import logging
import math

import higher
import torch
from torch import nn, optim
from tqdm import tqdm

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets.utils
import models.utils
from models.base_models_ori import ReplayMemory, TransformerRLN, LinearPLN

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OML-Log')


class OML:

    def __init__(self, device, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.device = device

        self.rln = TransformerRLN(model_name=kwargs.get('model'),
                                  max_length=kwargs.get('max_length'),
                                  device=device)
        self.pln = LinearPLN(in_dim=768, out_dim=1, device=device)
        meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                      [p for p in self.pln.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=3)
        self.loss_fn = nn.BCEWithLogitsLoss()

        logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))

        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def save_model(self, model_path):
        checkpoint = {'rln': self.rln.state_dict(),
                      'pln': self.pln.state_dict()}
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.rln.load_state_dict(checkpoint['rln'])
        self.pln.load_state_dict(checkpoint['pln'])

    def evaluate(self, dataloader, updates, mini_batch_size, model_name="roberta"):
        self.rln.eval()
        self.pln.train()

        support_set = []
        for _ in range(updates):
            text, label, candidates = self.memory.read_batch(batch_size=mini_batch_size)
            support_set.append((text, label, candidates))

        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpln, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, label, candidates in support_set:
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,label,candidates)
                add_prefix_space = False
                if model_name == "roberta":
                    replicated_text = [ " ".join(_t) for _t in replicated_text ]
                    replicated_relations = [ " ".join(_t) for _t in replicated_relations ]
                    add_prefix_space = True
                input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)), add_prefix_space)
                _repr = self.rln(input_dict)
                output = fpln(_repr)
                targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                loss = self.loss_fn(output, targets)

                diffopt.step(loss)
                pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(true_labels.tolist())

            acc = models.utils.calculate_accuracy(task_predictions, task_labels)
            logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(np.mean(support_loss), acc))

            all_losses, all_predictions, all_labels = [], [], []

            for text, label, candidates in tqdm(dataloader):
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,label,candidates)
                with torch.no_grad():
                    add_prefix_space = False
                    if model_name == "roberta":
                        replicated_text = [ " ".join(_t) for _t in replicated_text ]
                        replicated_relations = [ " ".join(_t) for _t in replicated_relations ]
                        add_prefix_space = True
                    input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)), add_prefix_space)
                    _repr = self.rln(input_dict)
                    output = fpln(_repr)
                    targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                    loss = self.loss_fn(output, targets)

                loss = loss.item()
                pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(true_labels.tolist())

        acc = models.utils.calculate_accuracy(all_predictions, all_labels)
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(np.mean(all_losses), acc))
        return acc

    def training(self, train_datasets, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        model_name = kwargs.get('model')

        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // mini_batch_size
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 1)))
            replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)
        else:
            replay_freq = 0
            replay_steps = 0
        logger.info('Replay frequency: {}'.format(replay_freq))
        logger.info('Replay steps: {}'.format(replay_steps))

        concat_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = iter(data.DataLoader(concat_dataset, batch_size=mini_batch_size, shuffle=False,
                                                collate_fn=datasets.utils.rel_encode))
        pbar = tqdm(total=math.ceil(len(concat_dataset)/mini_batch_size ))

        episode_id = 0
        while True:

            self.inner_optimizer.zero_grad()
            support_loss, support_acc = [], []

            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpln, diffopt):

                # Inner loop
                support_set = []
                task_predictions, task_labels = [], []
                for _ in range(updates):
                    try:
                        text, label, candidates = next(train_dataloader)
                        pbar.update(1)
                        support_set.append((text, label, candidates))
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return

                for text, label, candidates in support_set:
                    replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,label,candidates)
                    add_prefix_space = False
                    if model_name == "roberta":
                        replicated_text = [ " ".join(_t) for _t in replicated_text ]
                        replicated_relations = [ " ".join(_t) for _t in replicated_relations ]
                        add_prefix_space = True
                    input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)), add_prefix_space)
                    _repr = self.rln(input_dict)
                    output = fpln(_repr)
                    targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                    loss = self.loss_fn(output, targets)

                    diffopt.step(loss)
                    pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(true_labels.tolist())
                    self.memory.write_batch(text, label, candidates)

                acc = models.utils.calculate_accuracy(task_predictions, task_labels)

#                 logger.info('Episode {} support set: Loss = {:.4f}, accuracy = {:.4f}'.format(episode_id + 1,
#                                                                                               np.mean(support_loss),
#                                                                                               acc))

                # Outer loop
                query_loss, query_acc = [], []
                query_set = []

                if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0:
                    for _ in range(replay_steps):
                        text, label, candidates = self.memory.read_batch(batch_size=mini_batch_size)
                        query_set.append((text, label, candidates))
                else:
                    try:
                        text, label, candidates = next(train_dataloader)
                        pbar.update(1)
                        query_set.append((text, label, candidates))
                        self.memory.write_batch(text, label, candidates)
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return

                for text, label, candidates in query_set:
                    replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,label,candidates)
                    add_prefix_space = False
                    if model_name == "roberta":
                        replicated_text = [ " ".join(_t) for _t in replicated_text ]
                        replicated_relations = [ " ".join(_t) for _t in replicated_relations ]
                        add_prefix_space = True
                    input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)), add_prefix_space)
                    _repr = self.rln(input_dict)
                    output = fpln(_repr)
                    targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                    loss = self.loss_fn(output, targets)

                    query_loss.append(loss.item())
                    pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)

                    acc = models.utils.calculate_accuracy(pred.tolist(), true_labels.tolist())
                    query_acc.append(acc)

                    # RLN meta gradients
                    rln_params = [p for p in self.rln.parameters() if p.requires_grad]
                    meta_rln_grads = torch.autograd.grad(loss, rln_params, retain_graph=True)
                    for param, meta_grad in zip(rln_params, meta_rln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                    # PLN meta gradients
                    pln_params = [p for p in fpln.parameters() if p.requires_grad]
                    meta_pln_grads = torch.autograd.grad(loss, pln_params)
                    pln_params = [p for p in self.pln.parameters() if p.requires_grad]
                    for param, meta_grad in zip(pln_params, meta_pln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                # Meta optimizer step
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

#                 logger.info('Episode {} query set: Loss = {:.4f}, accuracy = {:.4f}'.format(episode_id + 1,
#                                                                                             np.mean(query_loss),
#                                                                                             np.mean(query_acc)))

                episode_id += 1
        pbar.close()

    def testing(self, test_dataset, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        model_name = kwargs.get('model')
        test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                          collate_fn=datasets.utils.rel_encode)
        acc = self.evaluate(dataloader=test_dataloader, updates=updates, mini_batch_size=mini_batch_size, model_name=model_name)
        logger.info('Overall test metrics: Accuracy = {:.4f}'.format(acc))
        return acc
