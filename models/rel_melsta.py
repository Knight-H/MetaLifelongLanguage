import logging
import math
import os, pickle

import higher
import torch
from torch import nn, optim
from tqdm import tqdm

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets.utils
import models.utils
from models.base_models_ori import ReplayMemory, TransformerRLN, LinearPLN, LabelAwareReplayMemory, TwoLinearPLN

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MeLSTA-Log')

class MeLSTA:
    def __init__(self, device, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.validation_split = kwargs.get('validation_split')
        self.task_aware = kwargs.get('task_aware')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.pln = kwargs.get('pln')
        self.task_dict = kwargs.get('task_dict', {})
        self.label2idx = kwargs.get('label2idx', {})
        self.idx2label = kwargs.get('idx2label', {})
        self.device = device

        self.rln = TransformerRLN(model_name=kwargs.get('model'),
                                  max_length=kwargs.get('max_length'),
                                  device=device)
        if self.pln == "1fc":
            self.pln = LinearPLN(in_dim=768, out_dim=1, device=device)
        elif self.pln == "2fc":
            self.pln = TwoLinearPLN(in_dim=768, out_dim=1, device=device)
        # For Task-aware
        self.memory = LabelAwareReplayMemory(write_prob=self.write_prob, tuple_size=3, n_classes=80, \
                                             validation_split=self.validation_split, task_dict=self.task_dict,\
                                             task_aware=self.task_aware)

        self.loss_fn = nn.BCEWithLogitsLoss()
            
        logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))
            
        meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                      [p for p in self.pln.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def save_model(self, model_path):
        checkpoint = {'rln': self.rln.state_dict(),
                      'pln': self.pln.state_dict()}
        torch.save(checkpoint, model_path)
        # Also save memory!!!
        pickle.dump( self.memory, open( model_path+"_memory.pickle", "wb" ), protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.rln.load_state_dict(checkpoint['rln'])
        self.pln.load_state_dict(checkpoint['pln'])
        with open(model_path+"_memory.pickle", 'rb') as f:
            self.memory = pickle.load(f)

    def evaluate(self, dataloader, updates, mini_batch_size, cluster_id, model_name="roberta"):
        self.rln.eval()
        self.pln.train()

        all_losses, all_predictions, all_labels = [], [], []

        # Get Query set first. and then find supporting support set
        for query_idx, (query_text, query_labels, query_candidates) in enumerate(dataloader):
            task_idx = cluster_id

            support_set = []
            for _ in range(updates):
                text, labels, candidates = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=task_idx, sort_score=True)
                # Change label from memory back to relation names
                _labels = [self.idx2label[label] for label in labels]
                support_set.append((text, _labels, candidates))

            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                    copy_initial_weights=False, track_higher_grads=False) as (fpln, diffopt):

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

                
                # Query set is now here!
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(query_text,
                                                                                query_labels,query_candidates)
                add_prefix_space = False
                if model_name == "roberta":
                    replicated_text = [ " ".join(_t) for _t in replicated_text ]
                    replicated_relations = [ " ".join(_t) for _t in replicated_relations ]
                    add_prefix_space = True
                with torch.no_grad():
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
        shuffle_index = kwargs.get('shuffle_index')
        model_name = kwargs.get('model')
        write_prob = kwargs.get('write_prob')

        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // mini_batch_size
            #replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 1)))
            #replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)
            # Calculation for Curriculum Replay
            # Average of (1-9)/9 = 5
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 5))) 
            #replay_steps = 5
            replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)
        else:
            replay_freq = 0
            replay_steps = 0
        logger.info('Replay frequency: {}'.format(replay_freq))
        logger.info('Replay steps: {}'.format(replay_steps))

         # DONT CONCAT! IT WILL STICK TGT.
        #concat_dataset = data.ConcatDataset(train_datasets)

        episode_id = 0
        
        for train_idx, train_dataset in enumerate(train_datasets):
            logger.info('Starting with train_idx: {}'.format(train_idx))
            task_idx = shuffle_index.index(train_idx) # Cluster number/ Task ID is the index of the shuffle!
            
            # Change to each dataset.
            train_dataloader = iter(data.DataLoader(data.ConcatDataset([train_dataset]), batch_size=mini_batch_size, shuffle=False,
                                                    collate_fn=datasets.utils.rel_encode))
            pbar = tqdm(total=math.ceil(len(train_dataset)/mini_batch_size ))
            is_break = False
            is_continue = False
            replay_task_idx = None
            is_replay = False
            replay_idx = 0
        
        
            while True:
                self.inner_optimizer.zero_grad()
                support_loss, support_acc = [], []
                
                # Reset replay everytime if it's not curriculum replay, else check the train_idx
                if is_replay and replay_task_idx == task_idx:
                    is_replay = False
                    replay_idx = 0
                    episode_id += 1 # else will forever loop replay!
                elif is_replay and replay_task_idx != task_idx:
                    print(f"replay_task_idx {replay_task_idx} task_idx {task_idx} replay_idx {replay_idx}")
                    is_replay = is_replay
                    replay_idx += 1
                    

                with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                          copy_initial_weights=False,
                                          track_higher_grads=False) as (fpln, diffopt):
                    # Inner loop
                    support_set = []
                    task_predictions, task_labels, task_label_confs = [], [], []
                    data_stream = []
                    # If Reverse Support and Task aware
                    # Get support from the ER Memory
                    # Edit: Pause any stream if it is replay!! (and replay steps!)
                    if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0 and episode_id > 1/write_prob*3:
                        is_replay = True
                        replay_task_idx = shuffle_index.index(replay_idx) # get replay_task_idx from replay_idx
                        for _ in range(replay_steps):
                            text, labels, candidates = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=replay_task_idx, \
                                                                       random_class=True)
                            # Change label from memory back to relation names
                            _labels = [self.idx2label[label] for label in labels]
                            support_set.append((text, _labels, candidates, [])) # fake indexes
                    else:
                        for _ in range(updates):
                            try:
                                text, labels, candidates = next(train_dataloader)
                                pbar.update(1)
                                data_stream.append((text, labels, candidates))
                                
                                text_support, labels_support, candidates_support, indexes_support, task_support = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=task_idx, with_index=True, no_support_priority=True, with_number_samples=True, random_class=True)
                                # Change label from memory back to relation names
                                labels_support = [self.idx2label[label] for label in labels_support]
                                #print(f"#Text Support {len(text_support)}")
                                #print(f"Text Support {text_support}")
                                #print(f"Task Support {task_support}")
                                #print(f"Labels Support {labels_support}")
                                #print(f"candidates_support {candidates_support}")
                                #print(f"indexes_support {indexes_support}")
                                #print(f"task_support {task_support}")
                                #print(f"self.memory.buffer_dict {self.memory.buffer_dict.keys()}")
                                if len(text_support) > 0:
                                    support_set.append((text_support, labels_support, candidates_support, indexes_support))
                                # If there is not enough task support on ending of batch, just do priority write and continue!
                                if task_support < mini_batch_size*updates*0.5:
                                    # Change the labels to relation idx
                                    _labels = [self.label2idx[" ".join(label)] for label in labels]
                                    self.memory.write_batch(text, _labels, candidates, task_id=task_idx, write_prob=1.)
                                    is_continue = True
                                else:
                                    # Change the labels to relation idx
                                    _labels = [self.label2idx[" ".join(label)] for label in labels]
                                    self.memory.write_batch(text, _labels, candidates, task_id=task_idx) # Normal write with write_prob=0.1
                                    is_continue = False
                            except StopIteration:
                                is_break = True
                                logger.info('Terminating training as all the data is seen')
                                break
                    # If end on priority write, just continue to next batch
                    if is_continue:
                        print("This is IS CONTINUE!")
                        is_continue=False
                        continue

                    for text, labels, candidates, indexes in support_set:
                        replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,labels,candidates)
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
                        pred, true_labels, label_conf = models.utils.make_rel_prediction(output, ranking_label, return_label_conf=True)
                        
                        support_loss.append(loss.item())
                        task_predictions.extend(pred.tolist())
                        task_labels.extend(true_labels.tolist())
                        task_label_confs.extend(label_conf.tolist())
                        
                        # Update the ER Scores
                        # Update 0->0', 1->1', 2->2',...
                        # Check on the same minibatch after adaptation. prev will be n (non-adapted) and curr will be a (adapted)
                        if not is_replay:
                            with torch.no_grad():
                                replicated_text_a, replicated_relations_a, ranking_label_a = datasets.utils.replicate_rel_data(text,labels,candidates)
                                add_prefix_space = False
                                if model_name == "roberta":
                                    replicated_text_a = [ " ".join(_t) for _t in replicated_text_a ]
                                    replicated_relations_a = [ " ".join(_t) for _t in replicated_relations_a ]
                                    add_prefix_space = True
                                input_dict_a = self.rln.encode_text(list(zip(replicated_text_a, replicated_relations_a)), add_prefix_space)
                                repr_a = self.rln(input_dict_a)
                                output_a = fpln(repr_a)
                                targets = torch.tensor(ranking_label_a).float().unsqueeze(1).to(self.device)
                                loss_a = self.loss_fn(output_a, targets)
                                pred_a, true_labels_a, label_conf_a = models.utils.make_rel_prediction(output_a.detach(), ranking_label_a, return_label_conf=True)
                                
                                # Things to Use: N and A
                                pred = pred.tolist()
                                label_ids = [self.label2idx[" ".join(label)] for label in labels]
                                label_conf = label_conf.tolist()
                                
                                pred_a = pred_a.tolist()
                                label_conf_a = label_conf_a.tolist()
                                
                                # Calculate a-n and update the memory
                                label_conf_diff = np.array(label_conf_a) - np.array(label_conf)
                                
                                # For training with random score (no sort score) - Comment line below!
                                self.memory.update_meta(label_ids, indexes, label_conf, label_conf_a, label_conf_diff, task_id=task_idx)

                    acc = models.utils.calculate_accuracy(task_predictions, task_labels)

    #                 logger.info('Episode {} support set: Loss = {:.4f}, accuracy = {:.4f}'.format(episode_id + 1,
    #                                                                                               np.mean(support_loss),
    #                                                                                               acc))

                    # Outer loop
                    query_loss, query_acc = [], []
                    query_set = []
                
                    if is_break:
                        break
                    # For Attempt #10, bring back the replay normal case.
                    # Edit: added is_replay for checking!
                    if is_replay:
                        for _ in range(replay_steps):
                            text, labels, candidates = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=replay_task_idx, \
                                                                       random_class=True)
                            # Change label from memory back to relation names
                            _labels = [self.idx2label[label] for label in labels]
                            query_set.append((text, _labels, candidates))
                    else:
                        # If we have reverse_support, just use data_stream as the query_set
                        query_set = data_stream


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

                    episode_id += 1 if not is_replay else 0
            pbar.close()

    def testing(self, test_datasets, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        model_name = kwargs.get('model')

        accuracies, precisions, recalls, f1s = [], [], [], []
        for cluster_id, test_dataset in enumerate(test_datasets):
            print('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                          collate_fn=datasets.utils.rel_encode)
            acc = self.evaluate(dataloader=test_dataloader, updates=updates, mini_batch_size=mini_batch_size,\
                                cluster_id=cluster_id, model_name=model_name)
            accuracies.append(acc)
        
        logger.info(f'[Acc] {accuracies}')
        logger.info('Overall test metrics: Accuracy = {:.4f}'.format(np.mean(accuracies)))
        return np.mean(accuracies)
