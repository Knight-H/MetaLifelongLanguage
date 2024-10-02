import logging
import os, pickle, time, csv
import torch
from torch import nn

import numpy as np

from torch.utils import data
import torch.nn.functional as F
from transformers import AdamW
from datetime import datetime

import datasets
import models.utils
#from models.base_models import TransformerClsModel, ReplayMemory
from models.base_models_ori import TransformerClsModel, ReplayMemory

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Replay-Log')


class Replay:

    def __init__(self, device, n_classes, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.device = device

        self.model = TransformerClsModel(model_name=kwargs.get('model'),
                                         n_classes=n_classes,
                                         max_length=kwargs.get('max_length'),
                                         device=device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        logger.info('Loaded {} as model'.format(self.model.__class__.__name__))

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def save_model(self, model_path):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

    def train(self, dataloader, n_epochs, log_freq, replay_freq, replay_steps, mini_batch_size):

        self.model.train()
        
        for epoch in range(n_epochs):
            all_losses, all_predictions, all_labels = [], [], []
            iter = 0

            for text, labels in dataloader:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()                

                if self.replay_rate != 0 and (iter + 1) % replay_freq == 0:
                    self.optimizer.zero_grad()
                    for _ in range(replay_steps):
                        ref_text, ref_labels = self.memory.read_batch(batch_size=mini_batch_size)
                        ref_labels = torch.tensor(ref_labels).to(self.device)
                        ref_input_dict = self.model.encode_text(ref_text)
                        ref_output = self.model(ref_input_dict)
                        ref_loss = self.loss_fn(ref_output, ref_labels)
                        ref_loss.backward()

                    params = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm(params, 25)
                    self.optimizer.step()

                loss = loss.item()
                pred = models.utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                iter += 1
                self.memory.write_batch(text, labels)

                if iter % log_freq == 0:
                    acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
                    logger.info(
                        'Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                        'F1 score = {:.4f}'.format(epoch + 1, np.mean(all_losses), acc, prec, rec, f1))
                    all_losses, all_predictions, all_labels = [], [], []

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels, all_label_conf = [], [], [], []

        self.model.eval()

        for text, labels in dataloader:
            labels = torch.tensor(labels).to(self.device)
            input_dict = self.model.encode_text(text)
            with torch.no_grad():
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)
            loss = loss.item()
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

    def training(self, train_datasets, **kwargs):
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 50)
        mini_batch_size = kwargs.get('mini_batch_size')
        model_dir = kwargs.get('model_dir')
        order = kwargs.get('order')
        learner = kwargs.get('learner')
        
        if self.replay_rate != 0:
            replay_freq = self.replay_every // mini_batch_size
            replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)
        else:
            replay_freq = 0
            replay_steps = 0
        logger.info('Replay frequency: {}'.format(replay_freq))
        logger.info('Replay steps: {}'.format(replay_steps))
        
        # Change this to dataset enumerate instead of Concat Dataset from OML-ER
        for train_idx, train_dataset in enumerate(train_datasets):
            logger.info('Training on train_idx {}'.format(train_idx))
            train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False,
                                               collate_fn=datasets.utils.batch_encode)
            self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq, replay_freq=replay_freq, replay_steps=replay_steps, mini_batch_size=mini_batch_size)
            #SAVE MODEL AND MEMORY EVERY EPOCH
            logger.info('Saving Model with train_idx: {}'.format(train_idx))
            _file_name = f"{learner.upper()[:3]}-order{order}-id{train_idx}-{str(datetime.now()).replace(':', '-').replace(' ', '_')}"
            MODEL_LOC = os.path.join(model_dir, _file_name + ".pt")
            MEMORY_SAVE_LOC = os.path.join(model_dir, _file_name + "_memory.pickle")
            self.save_model(MODEL_LOC)
            pickle.dump( self.memory, open( MEMORY_SAVE_LOC, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)

    def testing(self, test_datasets, **kwargs):
        tic = time.time()
        mini_batch_size = kwargs.get('mini_batch_size')
        model_path = kwargs.get('model_path')
        accuracies, precisions, recalls, f1s = [], [], [], []
        # Data for Visualization: [data_idx, label, label_conf, pred]
        data_for_visual = []
        
        for test_dataset in test_datasets:
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                              collate_fn=datasets.utils.batch_encode)
            acc, prec, rec, f1, all_pred, all_label, all_label_conf = self.evaluate(dataloader=test_dataloader)
            
            data_ids = [test_dataset.__class__.__name__ + str(i) for i in range(len(all_label))]
            data_for_visual.extend(list(zip(data_ids, all_label, all_label_conf, all_pred)))
            
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            
        _model_path0 = os.path.splitext(model_path)[0]
        csv_filename = _model_path0 +"_results.csv" 
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
        
