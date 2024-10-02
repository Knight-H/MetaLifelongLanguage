import logging, os, time, csv
import torch
from torch import nn
from tqdm import tqdm

import numpy as np

from torch.utils import data
import torch.nn.functional as F
from transformers import AdamW
from datetime import datetime

import datasets
import models.utils
# Change this to base_modesl_ori?
#from models.base_models import TransformerClsModel
from models.base_models_ori import TransformerClsModel

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Baseline-Log')


class Baseline:

    def __init__(self, device, n_classes, training_mode, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.device = device
        self.training_mode = training_mode
        self.model = TransformerClsModel(model_name=kwargs.get('model'),
                                         n_classes=n_classes,
                                         max_length=kwargs.get('max_length'),
                                         device=device)
        logger.info('Loaded {} as model'.format(self.model.__class__.__name__))
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def save_model(self, model_path):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

    # Minor Modification to make it accept early stopping
    # And also allow for it to save checkpoints on early stopping!
    def train(self, dataloader, n_epochs, log_freq, early_stopping=False, val_dataloader=None, model_dir=None):

        self.model.train()
        
        if early_stopping:
            n_epochs = 500 # Maximum to let it run
            early_stopper = EarlyStopper(patience=1)

        for epoch in range(n_epochs):
            logger.info(f"Starting Epoch {epoch+1}")
            all_losses, all_predictions, all_labels = [], [], []
            _iter = 0

            for text, labels in tqdm(dataloader):
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.model.encode_text(text)
                output = self.model(input_dict)
                loss = self.loss_fn(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                pred = models.utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())
                _iter += 1

                if _iter % log_freq == 0:
                    acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
                    logger.info(
                        'Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                        'F1 score = {:.4f}'.format(epoch + 1, np.mean(all_losses), acc, prec, rec, f1))
                    all_losses, all_predictions, all_labels = [], [], []
            
            
            #SAVE MODEL EVERY EPOCH
            logger.info('Saving Model on Epoch: {}'.format(epoch + 1))
            model_file_name = f"checkpoint-{epoch+1}.pt"
            MODEL_LOC = os.path.join(model_dir, model_file_name)
            self.save_model(MODEL_LOC)
            
            # For MTL Early Stopping, do Validation & Check val_loss
            if early_stopping:
                logger.info(f"Starting Validation...")
                _, _, _, _, val_all_pred, val_all_label, val_all_label_conf, val_loss = self.evaluate(dataloader=val_dataloader)
                logger.info(f"[VAL_LOSS] Epoch {epoch + 1}: {val_loss}")
                
                # Save validation loss as text in case logs are lost
                val_loss_file_name = "val_loss_" + model_file_name + ".txt"
                with open(os.path.join(model_dir, val_loss_file_name), 'w') as f:
                    f.write(str(val_loss))

                if early_stopper.early_stop(val_loss):             
                    logger.info(f"Stopping from Early Stopping... min_loss {early_stopper.min_validation_loss}")
                    break
                

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels, all_label_conf = [], [], [], []

        self.model.eval()

        for text, labels in tqdm(dataloader):
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

        return acc, prec, rec, f1, all_predictions, all_labels, all_label_conf, np.mean(all_losses)

    def training(self, train_datasets, val_datasets, **kwargs):
        tic = time.time()
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 500)
        mini_batch_size = kwargs.get('mini_batch_size')
        model_dir = kwargs.get('model_dir')
        order = kwargs.get('order')
        learner = kwargs.get('learner')
        # For Early Stopping 
        early_stopping = kwargs.get('early_stopping', False)

        if self.training_mode == 'sequential':
            for train_idx, train_dataset in enumerate(train_datasets):
                logger.info('Training on train_idx {}'.format(train_idx))
                train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False,
                                                   collate_fn=datasets.utils.batch_encode)
                self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)
                #SAVE MODEL AND MEMORY EVERY EPOCH
                logger.info('Saving Model with train_idx: {}'.format(train_idx))
                model_file_name = f"{learner.upper()[:3]}-order{order}-id{train_idx}-{str(datetime.now()).replace(':', '-').replace(' ', '_')}.pt"
                MODEL_LOC = os.path.join(model_dir, model_file_name)
                self.save_model(MODEL_LOC)

        elif self.training_mode == 'multi_task':
            # Load both train & validation!
            train_dataset = data.ConcatDataset(train_datasets)
            val_dataset = data.ConcatDataset(val_datasets)
            logger.info('Training multi-task model on all datasets')
            train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True,
                                               collate_fn=datasets.utils.batch_encode)
            
            val_dataloader = data.DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=True,
                                               collate_fn=datasets.utils.batch_encode)
            self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq, 
                       early_stopping=early_stopping, val_dataloader=val_dataloader, model_dir=model_dir)
        else:
            raise ValueError('Invalid training mode')
            
        toc = time.time() - tic
        logger.info(f"Total Training Time used: {toc//60} minutes")

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
            acc, prec, rec, f1, all_pred, all_label, all_label_conf, _ = self.evaluate(dataloader=test_dataloader)
            
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
        

# Since only MTL uses Early Stopping, I'll just be pasting it here!
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

