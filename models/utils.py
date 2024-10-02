import torch
from torch.nn import functional as F
import numpy as np
from sklearn import metrics


def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = np.array(predictions)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    return accuracy, precision, recall, f1_score


def calculate_accuracy(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    return accuracy


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


def make_rel_prediction(cosine_sim, ranking_label, return_label_conf=False):
    pred, label_confs = [], []
    with torch.no_grad():
        pos_idx = [i for i, lbl in enumerate(ranking_label) if lbl == 1]
        if len(pos_idx) == 1:
            pred.append(torch.argmax(cosine_sim))
        else:
            for i in range(len(pos_idx)): # Somehow the old code is len(pos_idx) - 1, this means you always skip the last answer?!?
                start_idx = pos_idx[i]
                end_idx = pos_idx[i+1] if i < len(pos_idx)-1 else len(ranking_label)  
                subset = cosine_sim[start_idx: end_idx]
                output_softmax = F.softmax(subset, -1)
                label_conf = output_softmax[0]
                pred.append(torch.argmax(subset))
                label_confs.append(label_conf)
    pred = torch.tensor(pred)            # whatever is the argmax of the subset
    true_labels = torch.zeros_like(pred) # will always be the first one
    label_confs = torch.tensor(label_confs) 
    
    if return_label_conf:
        return pred, true_labels, label_confs
    return pred, true_labels


def split_rel_scores(cosine_sim, ranking_label):
    pos_scores, neg_scores = [], []
    pos_index = 0
    for i in range(len(ranking_label)):
        if ranking_label[i] == 1:
            pos_index = i
        elif ranking_label[i] == -1:
            pos_scores.append(cosine_sim[pos_index])
            neg_scores.append(cosine_sim[i])
    pos_scores = torch.stack(pos_scores)
    neg_scores = torch.stack(neg_scores)
    return pos_scores, neg_scores
