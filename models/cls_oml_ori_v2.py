import logging
import math, random
import os, pickle, csv, json, time
from tqdm import tqdm

import higher
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data

import numpy as np

from transformers import AdamW
from datetime import datetime

import models.utils
import datasets.utils
from models.base_models_ori import TransformerRLN, LinearPLN, ReplayMemory, LabelAwareReplayMemory, TwoLinearPLN

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OML-Log')


# dataclass_mapper = {
#     "AGNewsDataset": 0,
#     "AmazonDataset": 1,
#     "YelpDataset": 2,
#     "DBPediaDataset": 3,
#     "YahooAnswersDataset": 4
# }

# Define the ordering of the datasets
dataset_order_mapping = {
    1: [2, 0, 3, 1, 4],
    2: [3, 4, 0, 1, 2],
    3: [2, 4, 1, 3, 0],
    4: [0, 2, 1, 4, 3]
}

class OML:

    def __init__(self, device, n_classes, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.validation_split = kwargs.get('validation_split')
        self.task_aware = kwargs.get('task_aware')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.pln = kwargs.get('pln')
        self.device = device

        self.rln = TransformerRLN(model_name=kwargs.get('model'),
                                  max_length=kwargs.get('max_length'),
                                  device=device)
        if self.pln == "1fc":
            self.pln = LinearPLN(in_dim=768, out_dim=n_classes, device=device)
        elif self.pln == "2fc":
            self.pln = TwoLinearPLN(in_dim=768, out_dim=n_classes, device=device)
        # If it is selective replay
        if kwargs.get('selective_replay'):
            # For Task-aware
            task_dict = {}
            if self.task_aware:
                task_dict = {
                    0: list(range(5, 9)), # AG
                    1: list(range(0, 5)), # Amazon
                    2: list(range(0, 5)), # Yelp
                    3: list(range(9, 23)), # DBPedia
                    4: list(range(23, 33)), # Yahoo
                }
            self.memory = LabelAwareReplayMemory(write_prob=self.write_prob, tuple_size=2, n_classes=n_classes, \
                                                 validation_split=self.validation_split, task_dict=task_dict, task_aware=self.task_aware, \
                                                 filter_support=kwargs.get('reverse_support', False)) # only get filter condition on D_S <-- Mem, else no support > 1!
        else:
            self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.loss_fn = nn.CrossEntropyLoss()

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

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.rln.load_state_dict(checkpoint['rln'])
        self.pln.load_state_dict(checkpoint['pln'])

    def evaluate(self, dataloader, updates, mini_batch_size):
        self.rln.eval()
        self.pln.train()

        support_set = []
        for _ in range(updates):
            text, labels = self.memory.read_batch(batch_size=mini_batch_size)
            support_set.append((text, labels))

        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpln, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.rln.encode_text(text)
                _repr = self.rln(input_dict)
                output = fpln(_repr)
                loss = self.loss_fn(output, labels)
                diffopt.step(loss)
                pred = models.utils.make_prediction(output.detach())
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(labels.tolist())

            acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

#             logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
#                         'recall = {:.4f}, F1 score = {:.4f}'.format(np.mean(support_loss), acc, prec, rec, f1))

            all_losses, all_predictions, all_labels, all_label_conf = [], [], [], []

            for text, labels in tqdm(dataloader):
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.rln.encode_text(text)
                with torch.no_grad():
                    _repr = self.rln(input_dict)
                    output = fpln(_repr)
                    loss = self.loss_fn(output, labels)
                loss = loss.item()
                # output.detach().max(-1) max on each Batch, which will return [0] max, [1] indices
                output_softmax = F.softmax(output, -1)
                label_conf = output_softmax[np.arange(len(output_softmax)), labels] # Select labels in the softmax of 33 classes

                pred = models.utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                all_label_conf.extend(label_conf.tolist())

        acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))

        return acc, prec, rec, f1, all_predictions, all_labels, all_label_conf
    
    

    def training(self, train_datasets, val_datasets, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        order = kwargs.get('order')
        task_aware = kwargs.get('task_aware')
        selective_replay = kwargs.get('selective_replay')
        reverse_support = kwargs.get('reverse_support')
        # all_query_replay
        all_query_replay = kwargs.get('all_query_replay')
        write_prob = kwargs.get('write_prob')
        curriculum_replay = kwargs.get('curriculum_replay')

        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // mini_batch_size
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 1)))
            # Average of 1-4 = 2.5
            #replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 2.5))) 
            # Edit: calculate this outside!
            replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)
            #replay_steps = 5
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
            print(train_dataset.__class__)
            #task_idx = dataclass_mapper[train_dataset.__class__.__name__]
            task_idx = dataset_order_mapping[order][train_idx]
            # Change to each dataset.
            train_dataloader = iter(data.DataLoader(data.ConcatDataset([train_dataset]), batch_size=mini_batch_size, shuffle=False,
                                                    collate_fn=datasets.utils.batch_encode))
            pbar = tqdm(total=math.ceil(len(train_dataset)/mini_batch_size ))
            is_break = False
            is_continue = False
            replay_task_idx = None
            is_replay = False
            replay_idx = 0
            
            while True:
                self.inner_optimizer.zero_grad()
                support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []
                
                # Reset replay everytime if it's not curriculum replay, else check the train_idx
                if curriculum_replay:
                    if is_replay:
                        print(f"replay_task_idx {replay_task_idx} task_idx {task_idx} replay_idx {replay_idx}")
                    if is_replay and replay_task_idx == task_idx:
                        is_replay = False
                        replay_idx = 0
                        episode_id += 1 # else will forever loop replay!
                    elif is_replay and replay_task_idx != task_idx:
                        is_replay = is_replay
                        replay_idx += 1
                else:
                    # For cases without curriculum_replay, don't forget to make is_replay false!!
                    if is_replay:
                        is_replay = False
                        episode_id += 1 # else will forever loop replay!


                with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                          copy_initial_weights=False,
                                          track_higher_grads=False) as (fpln, diffopt):
                    # Inner loop
                    support_set = []
                    task_predictions, task_labels, task_label_confs = [], [], []
                    # Case Reverse Support,  D_S <-- Mem (Either query on task OR replay)
                    # - If Reverse Support and Task aware: Get support from the TAM Memory (Read Batch Task)
                    # - Else normal                      : Get support from the ER Memory (Read Batch)
                    # Else normal,          D_S <-- SplitDS
                    if reverse_support:
                        data_stream = []
                        # Pause any stream if it is replay!! (and replay steps!)
                        if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0 and episode_id > 1/write_prob*3:
                            is_replay = True
                            if task_aware and selective_replay:
                                # For curriculum replay: get replay_task_idx from replay_idx
                                # - Else: random task replay_task_idx
                                replay_task_idx = dataset_order_mapping[order][replay_idx] if curriculum_replay \
                                                else random.choice(dataset_order_mapping[order][:train_idx+1])
                            for _ in range(replay_steps):
                                if task_aware and selective_replay:
                                    text, labels = self.memory.read_batch_task(batch_size=mini_batch_size, \
                                                                        task_idx=replay_task_idx, random_class=True)
                                else:
                                    text, labels = self.memory.read_batch(batch_size=mini_batch_size)
                                support_set.append((text, labels, [])) # fake indexes
                        else:
                            for _ in range(updates):
                                try:
                                    text, labels  = next(train_dataloader)
                                    pbar.update(1)
                                    data_stream.append((text, labels))

                                    if task_aware and selective_replay:
                                        text_support, labels_support, indexes_support, task_support = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=task_idx, with_index=True, no_support_priority=True, with_number_samples=True, random_class=True)
                                        # [A.1] For training with random score (no sort score)
                                        #text_support, labels_support, indexes_support, task_support = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=task_idx, with_index=True, no_support_priority=False, with_number_samples=True, random_class=True)
                                    else: 
                                        text_support, labels_support = self.memory.read_batch(batch_size=mini_batch_size)
                                        indexes_support = []             # fake indexes
                                        task_support = len(self.memory)  # Length of Buffer
                                    # This may yield empty if it's the first
                                    if len(text_support) > 0:
                                        support_set.append((text_support, labels_support, indexes_support))
                                    # If there is not enough task support on ending of batch, just do priority write and continue!
                                    if task_support < mini_batch_size*updates*0.5:
                                        self.memory.write_batch(text, labels, task_id=task_idx, write_prob=1.) if task_aware and selective_replay else self.memory.write_batch(text, labels, write_prob=1.)
                                        is_continue = True
                                    else:
                                        self.memory.write_batch(text, labels, task_id=task_idx) if task_aware and selective_replay else self.memory.write_batch(text, labels) # Normal write with write_prob=0.1
                                        is_continue = False
                                except StopIteration:
                                    is_break = True
                                    logger.info('Terminating training as all the data is seen')
                                    break
                    else:
                        # If normal cases, D_S <-- SplitDS
                        for _ in range(updates):
                            try:
                                text, labels = next(train_dataloader)
                                pbar.update(1)
                                if task_aware and selective_replay:
                                    indexes_support = self.memory.write_batch(text, labels, task_id=task_idx, with_index=True)
                                    # print("Index support ",indexes_support )
                                else:
                                    self.memory.write_batch(text, labels) # Changed writing batch to here instead
                                    indexes_support = [] # fake indexes
                                support_set.append((text, labels, indexes_support)) 
                            except StopIteration:
                                is_break = True
                                # print('Terminating training as all the data is seen')
                                logger.info('Terminating training as all the data is seen')
                                break
                    # If end on priority write, just continue to next batch
                    if is_continue:
                        print("This is IS CONTINUE!")
                        is_continue=False
                        continue
                    # print("TOTAL INDEXES ", [x[2] for x in support_set])

                    for text, labels, indexes in support_set:
                        #print(f"TEXT {text}")
                        #print(f"labels {labels}")
                        labels = torch.tensor(labels).to(self.device)
                        input_dict = self.rln.encode_text(text)
                        _repr = self.rln(input_dict)
                        output = fpln(_repr)
                        loss = self.loss_fn(output, labels)
                        diffopt.step(loss)
                        output_softmax = F.softmax(output, -1)
                        label_conf = output_softmax[np.arange(len(output_softmax)), labels] # Select labels in the softmax of 33 classes
                        pred = models.utils.make_prediction(output.detach())
                        
                        support_loss.append(loss.item())
                        task_predictions.extend(pred.tolist())
                        task_labels.extend(labels.tolist())
                        task_label_confs.extend(label_conf.tolist())
                    
                        # Update the ER Scores
                        # Update 0->0', 1->1', 2->2',...
                        # Check on the same minibatch after adaptation. prev will be n (non-adapted) and curr will be a (adapted)
                        if task_aware and selective_replay and not is_replay:
                            with torch.no_grad():
                                input_dict_a = self.rln.encode_text(text)
                                repr_a = self.rln(input_dict_a)
                                output_a = fpln(repr_a) # Output has size of torch.Size([16, 33]) [BATCH, CLASSES]
                                loss_a = self.loss_fn(output_a, labels)
                                # output.detach().max(-1) max on each Batch, which will return [0] max, [1] indices
                                output_softmax_a = F.softmax(output_a, -1)
                                label_conf_a = output_softmax_a[np.arange(len(output_softmax_a)), labels] # Select labels in the softmax of 33 classes
                                pred_a = models.utils.make_prediction(output_a.detach())
                                
                                
                                # Things to Use: N and A
                                pred = pred.tolist()
                                labels = labels.tolist()
                                label_conf = label_conf.tolist()
                                
                                pred_a = pred_a.tolist()
                                label_conf_a = label_conf_a.tolist()
                                
                                # Calculate a-n and update the memory
                                label_conf_diff = np.array(label_conf_a) - np.array(label_conf)
                                # [A.1] For training with random score (no sort score) - Comment line below!
                                self.memory.update_meta(labels, indexes, label_conf, label_conf_a, label_conf_diff, task_id=task_idx)

                    acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

#                     logger.info('Episode {} support set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
#                                 'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
#                                                                             np.mean(support_loss), acc, prec, rec, f1))

                    # Outer loop
                    query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                    query_set = []
                    
                    if is_break:
                        break

                    # Case Reverse Support,  D_Q <-- DS/Rep
                    # Case All Query,        D_Q <-- Mem
                    # Else Normal,           D_Q <-- SplitDS/Rep
                    elif reverse_support:
                        # For Attempt #10, bring back the replay normal case.
                        # Edit: added is_replay for checking!
                        if is_replay:
                            for _ in range(replay_steps):
                                if task_aware and selective_replay:
                                    # For reverse support, replay_task_idx is already got from above ie. D_S <-- Mem and D_Q <-- Mem
                                    text, labels = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=replay_task_idx, \
                                                                           random_class=True)
                                else:
                                    text, labels = self.memory.read_batch(batch_size=mini_batch_size)
                                query_set.append((text, labels))
                        else:
                            # If we have reverse_support, just use data_stream as the query_set
                            query_set = data_stream
                    elif all_query_replay:
                        # Should i have more ?? MAML-CL Reads all query set of 45!!
                        text, labels = self.memory.read_batch(batch_size=mini_batch_size)
                        query_set.append((text, labels))
                    else:
                        # Normal Case
                        if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0:
                            for _ in range(replay_steps):
                                if task_aware and selective_replay:
                                    # For curriculum replay: get replay_task_idx from replay_idx
                                    # - Else: random task replay_task_idx
                                    replay_task_idx = dataset_order_mapping[order][replay_idx] if curriculum_replay \
                                                    else random.choice(dataset_order_mapping[order][:train_idx+1])
                                    text, labels = self.memory.read_batch_task(batch_size=mini_batch_size, task_idx=replay_task_idx, \
                                                                           random_class=True)
                                else:
                                    text, labels = self.memory.read_batch(batch_size=mini_batch_size)
                                query_set.append((text, labels))
                        else:
                            try:
                                text, labels = next(train_dataloader)
                                pbar.update(1)
                                query_set.append((text, labels))
                                # This is writing batch. But we're not updating indexes here because there's no adaptation. So 
                                # OML-ER with task-aware and selective-replay will have some WITHOUT support
                                self.memory.write_batch(text, labels, task_id=task_idx) if task_aware and selective_replay else self.memory.write_batch(text, labels)
                            except StopIteration:
                                logger.info('Terminating training as all the data is seen')
                                break

                    for text, labels in query_set:
                        labels = torch.tensor(labels).to(self.device)
                        input_dict = self.rln.encode_text(text)
                        _repr = self.rln(input_dict)
                        output = fpln(_repr)
                        loss = self.loss_fn(output, labels)
                        query_loss.append(loss.item())
                        pred = models.utils.make_prediction(output.detach())

                        acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())
                        query_acc.append(acc)
                        query_prec.append(prec)
                        query_rec.append(rec)
                        query_f1.append(f1)

                        # RLN meta gradients
                        rln_params = [p for p in self.rln.parameters() if p.requires_grad]
                        meta_rln_grads = torch.autograd.grad(loss, rln_params, retain_graph=True)
                        for param, meta_grad in zip(rln_params, meta_rln_grads):
                            if param.grad is not None:
                                param.grad += meta_grad.detach()
                            else:
                                param.grad = meta_grad.detach()

                        # # PLN meta gradients
                        pln_params = [p for p in fpln.parameters() if p.requires_grad]
                        meta_pln_grads = torch.autograd.grad(loss, pln_params)
                        pln_params = [p for p in self.pln.parameters() if p.requires_grad]
                        for param, meta_grad in zip(pln_params, meta_pln_grads):
                            if param.grad is not None:
                                param.grad += meta_grad.detach()
                            else:
                                param.grad = meta_grad.detach()
                        #loss.backward()

                    # Meta optimizer step
                    self.meta_optimizer.step()
                    self.meta_optimizer.zero_grad()

#                     logger.info('Episode {} query set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
#                                 'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
#                                                                             np.mean(query_loss), np.mean(query_acc),
#                                                                             np.mean(query_prec), np.mean(query_rec),
#                                                                             np.mean(query_f1)))
                    episode_id += 1 if not is_replay else 0
            pbar.close()
            #SAVE MODEL AND MEMORY EVERY EPOCH
            logger.info('Saving Model with train_idx: {}'.format(train_idx))
            model_dir = "/data/model_runs/original_oml"
            model_file_name = f"{self.__class__.__name__}-order{order}-id{train_idx}-{str(datetime.now()).replace(':', '-').replace(' ', '_')}.pt"
            MODEL_LOC = os.path.join(model_dir, model_file_name)
            MEMORY_SAVE_LOC = os.path.join(model_dir, f"{self.__class__.__name__}-order{order}-id{train_idx}-{str(datetime.now()).replace(':', '-').replace(' ', '_')}_memory.pickle")
            self.save_model(MODEL_LOC)
            pickle.dump( self.memory, open( MEMORY_SAVE_LOC, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)

    def validate(self, fpln, rln, validation_set):
        all_valid_preds, all_valid_labels, all_valid_label_conf = [], [], []

        for valid_text, valid_labels, _ in validation_set:        
            valid_labels = torch.tensor(valid_labels).to(device)
            valid_input_dict = learner.rln.encode_text(valid_text)
            valid_repr = learner.rln(valid_input_dict)
            valid_output = fpln(valid_repr) # Output has size of torch.Size([16, 33]) [BATCH, CLASSES]
            valid_loss = learner.loss_fn(valid_output, valid_labels)
            valid_loss = valid_loss.item()

            # output.detach().max(-1) max on each Batch, which will return [0] max, [1] indices
            valid_output_softmax = F.softmax(valid_output, -1)
            valid_label_conf = valid_output_softmax[np.arange(len(valid_output_softmax)), valid_labels] # Select labels in the softmax of 33 classes

            valid_pred = models.utils.make_prediction(valid_output.detach())


            # Put in things to return
            # all_valid_losses.extend(valid_loss)
            all_valid_preds.extend(valid_pred.tolist())
            all_valid_labels.extend(valid_labels.tolist())
            all_valid_label_conf.extend(valid_label_conf.tolist())
        return all_valid_preds, all_valid_labels, all_valid_label_conf # removed loss, since no need

    def testing(self, test_datasets, **kwargs):
        tic = time.time()
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        model_path = kwargs.get('model_path')
        logger.info('----------Testing on test set starts here----------')
        
        accuracies, precisions, recalls, f1s = [], [], [], []
        # Data for Visualization: [data_idx, label, label_conf, pred]
        data_for_visual = []
        for test_dataset in test_datasets:
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                              collate_fn=datasets.utils.batch_encode)
            acc, prec, rec, f1, all_pred, all_label, all_label_conf = self.evaluate(dataloader=test_dataloader, updates=updates, mini_batch_size=mini_batch_size)
            
            data_ids = [test_dataset.__class__.__name__ + str(i) for i in range(len(all_label))]
            data_for_visual.extend(list(zip(data_ids, all_label, all_label_conf, all_pred)))
            
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        _model_path0 = os.path.splitext(model_path)[0]
        csv_filename = _model_path0 + "_update"+ str(updates) +"_results_sr.csv" # for selective replay
        with open(csv_filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["data_idx", "label", "label_conf", "pred"])
            csv_writer.writerows(data_for_visual)
        logger.info(f"Done writing CSV File at {csv_filename}")
            
        logger.info("COPY PASTA - not really but ok")
        for row in accuracies:
            logger.info(row)
        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))

        toc = time.time() - tic
        logger.info(f"Total Time used: {toc//60} minutes")
        return accuracies
    

