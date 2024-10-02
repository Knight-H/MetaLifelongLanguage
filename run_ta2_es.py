#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random, gc, os, pickle, csv, time, json 

import datasets.utils
import models.utils
from models.cls_oml_ori_v2 import OML
from models.base_models_ori import LabelAwareReplayMemory

import numpy as np

import higher
import torch
import torch.nn.functional as F
from torch.utils import data


# # Constants

# In[2]:


dataset_order_mapping = {
    1: [2, 0, 3, 1, 4],
    2: [3, 4, 0, 1, 2],
    3: [2, 4, 1, 3, 0],
    4: [0, 2, 1, 4, 3]
}
n_classes = 33
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# model_path = "/data/model_runs/original_oml/aOML-order1-2022-07-18/OML-order1-id4-2022-07-18_17-53-13.518612.pt"
# model_path = "/data/model_runs/original_oml/aOML-order1-inlr002-2022-07-31/OML-order1-id4-2022-07-31_14-53-46.456804.pt"
# model_path = "/data/model_runs/original_oml/aOML-order1-inlr005-2022-07-31/OML-order1-id4-2022-07-31_18-47-41.477968.pt"
# model_path = "/data/model_runs/original_oml/aOML-order1-inlr005-up20-2022-08-01/OML-order1-id4-2022-08-01_14-45-55.869765.pt"
# model_path = "/data/model_runs/original_oml/aOML-order1-inlr010-2022-07-31/OML-order1-id4-2022-07-31_21-18-36.241546.pt"
# model_path = "/data/model_runs/original_oml/aOML-order1-inlr020-2022-08-16/OML-order1-id4-2022-08-16_11-37-19.424113.pt"
# model_path = "/data/model_runs/original_oml/aOML-order1-inlr050-2022-08-16/OML-order1-id4-2022-08-16_14-16-12.167637.pt"

# v. SR 
# model_path = "/data/model_runs/original_oml/aOML-order1-inlr010-2022-08-29-sr/OML-order1-id4-2022-08-29_18-10-31.695669.pt"
# v. SR Query
model_path = "/data/model_runs/original_oml/aOML-order1-inlr010-2022-08-30-sr-query/OML-order1-id4-2022-08-30_05-21-18.854228.pt"

# memory_path = "/data/model_runs/original_oml/aOML-order1-2022-07-18/OML-order1-id4-2022-07-18_17-53-13.518639_memory.pickle"
# memory_path = "/data/model_runs/original_oml/aOML-order1-inlr002-2022-07-31/OML-order1-id4-2022-07-31_14-53-46.456828_memory.pickle"
# memory_path = "/data/model_runs/original_oml/aOML-order1-inlr005-2022-07-31/OML-order1-id4-2022-07-31_18-47-41.477992_memory.pickle"
# memory_path = "/data/model_runs/original_oml/aOML-order1-inlr005-up20-2022-08-01/OML-order1-id4-2022-08-01_14-45-55.869797_memory.pickle"
# memory_path = "/data/model_runs/original_oml/aOML-order1-inlr010-2022-07-31/OML-order1-id4-2022-07-31_21-18-36.241572_memory.pickle"
# memory_path = "/data/model_runs/original_oml/aOML-order1-inlr020-2022-08-16/OML-order1-id4-2022-08-16_11-37-19.424139_memory.pickle"
# memory_path = "/data/model_runs/original_oml/aOML-order1-inlr050-2022-08-16/OML-order1-id4-2022-08-16_14-16-12.167666_memory.pickle"
# v. SR 
# memory_path = "/data/model_runs/original_oml/aOML-order1-inlr010-2022-08-29-sr/OML-order1-id4-2022-08-29_18-10-31.695692_memory.pickle"
# v. SR Query
memory_path = "/data/model_runs/original_oml/aOML-order1-inlr010-2022-08-30-sr-query/OML-order1-id4-2022-08-30_05-21-18.854254_memory.pickle"


# new_memory_path, ext = os.path.splitext(memory_path)
# new_memory_path = new_memory_path + "_label" + ext

use_db_cache = True
cache_dir = 'tmp'


# In[3]:


args = {
    "order": 1,
    "n_epochs": 1,
    "lr": 3e-5,
    "inner_lr": 0.001*10,
    "meta_lr": 3e-5,
    "model": "bert",
    "learner": "oml",
    "mini_batch_size": 16,
    "updates": 5*2,
    "write_prob": 1.0,
    "max_length": 448,
    "seed": 42,
    "replay_rate": 0.01,
    "replay_every": 9600
}
updates = args["updates"]
mini_batch_size = args["mini_batch_size"]
order = args["order"]


# In[4]:


torch.manual_seed(args["seed"])
random.seed(args["seed"])
np.random.seed(args["seed"])


# # Load Dataset

# In[5]:


print('Loading the datasets')
test_datasets = []
for dataset_id in dataset_order_mapping[order]:
    test_dataset_file = os.path.join(cache_dir, f"{dataset_id}.cache")
    if os.path.exists(test_dataset_file):
        with open(test_dataset_file, 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        test_dataset = datasets.utils.get_dataset_test("", dataset_id)
        print('Loaded {}'.format(test_dataset.__class__.__name__))
        test_dataset = datasets.utils.offset_labels(test_dataset)
        pickle.dump(test_dataset, open( test_dataset_file, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Pickle saved at {test_dataset_file}")
    test_datasets.append(test_dataset)
print('Finished loading all the datasets')


# In[6]:


import random
buffer_dict = {
    0: [1,2,3],
    1: [2,3,4]
}


buffer_dict_valid = {} # Reset the buffer_dict_valid
for class_idx, buffer_list in buffer_dict.items():
    buffer_dict_valid[class_idx] = []
    for i in range(2):
        pop_data = buffer_dict[class_idx].pop(random.randint(0,len(buffer_dict[class_idx])-1))
        buffer_dict_valid[class_idx].append(pop_data)
print(buffer_dict)
print(buffer_dict_valid)


# # Load Model

# In[7]:


learner = OML(device=device, n_classes=n_classes, **args)
print('Using {} as learner'.format(learner.__class__.__name__))
learner.load_model(model_path)
with open(memory_path, 'rb') as f:
#     learner.memory = pickle.load(f)
    memory_buffer = pickle.load(f)


# In[8]:


# Setting up task dict for task-aware
memory_buffer.task_dict = {
    0: list(range(5, 9)), # AG
    1: list(range(0, 5)), # Amazon
    2: list(range(0, 5)), # Yelp
    3: list(range(9, 23)), # DBPedia
    4: list(range(23, 33)), # Yahoo
}


# In[9]:


dataclass_mapper = {
    "AGNewsDataset": 0,
    "AmazonDataset": 1,
    "YelpDataset": 2,
    "DBPediaDataset": 3,
    "YahooAnswersDataset": 4
}
dataclass_mapper["AGNewsDataset"]


# In[10]:


# For hacking the early stopping to work!!
# Split the buffer_dict (train) to buffer_dict_valid (validation)
memory_buffer.split_train_validation()


# In[11]:


len(memory_buffer.buffer_dict[0])


# In[12]:


len(memory_buffer.buffer_dict_valid[0])


# # Testing
# 
# Select specific column index per row
# https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes

# In[13]:


# Returns loss,preds,labels, labels_conf
def validate(fpln, validation_set, batch_size = 16):
    all_valid_losses, all_valid_preds, all_valid_labels, all_valid_label_conf = [], [], [], []
    
    # https://stackoverflow.com/questions/2231663/slicing-a-list-into-a-list-of-sub-lists
    for i in range(0, len(validation_set[1]), batch_size):
        valid_text = validation_set[0][i:i+batch_size]
        valid_labels = validation_set[1][i:i+batch_size]
        
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
    
# Compare diff results between the unadapted vs adapted
# Returns Dictionary of class_idx -> [ a - n, ...  ] for each i (300). Can np.sum() or np.mean() later
# validate_labels = The labels (Shared)
# validate_label_conf_0 = The label conf of validate_0
# validate_label_conf_n = The label conf of validate_n
def calculate_diff_class(validate_labels, validate_label_conf_0, validate_label_conf_n): 
    # Adapted confs - NonAdapted Confs (a-n)
    validate_label_conf_diff = np.array(validate_label_conf_n) - np.array(validate_label_conf_0)
    
    # The dictionary to return  class_idx -> [ a - n, ...  ] 
    return_dict = {}
    for i, class_idx in enumerate(validate_labels):
        # Filter conf_diff by class
        return_dict[class_idx] = return_dict.get(class_idx, []) + [validate_label_conf_diff[i]]
    return return_dict


# In[14]:


def evaluate(dataloader, updates, mini_batch_size, dataname=""):
    learner.rln.eval()
    learner.pln.train()
    
    all_losses, all_predictions, all_labels, all_label_conf = [], [], [], []
    
    # Map of update# --> array<item>
    # THIS IS NOT CORRECT ! WILL GET ONLY 1 QUERY IDX!
    valid_losses, valid_preds, valid_labels, valid_label_conf = {}, {}, {}, {}
    valid_class_diff = {}
    
    all_adaptation_time = []
    # Get Query set first. and then find supporting support set
    for query_idx, (query_text, query_labels) in enumerate(dataloader):
        
        # Only get the first 20 queries! (as sample to plot)
        if (query_idx > 10):
            continue
        
        print(f"Query ID {query_idx}/{len(dataloader)}")
        # The task id to optimize to for support set
        # task_idx = get_task_from_label_list(query_labels, memory_buffer.task_dict)
        task_idx = dataclass_mapper[dataname]
        
    
        support_set = []
        for _ in range(updates):
            text, labels = memory_buffer.read_batch_task(batch_size=mini_batch_size, task_idx=task_idx)
            support_set.append((text, labels))
        

        # Get Validation Set of the task_idx - nah
        # Get Validation Set of the whole set
        validation_set = memory_buffer.read_batch_validation()
        
        with higher.innerloop_ctx(learner.pln, learner.inner_optimizer,
                                  copy_initial_weights=False, track_higher_grads=False) as (fpln, diffopt):
            
            # Test validation_set BEFORE the update (update=0)
            with torch.no_grad():
                #validation_results_0 = validate(fpln, validation_set, batch_size=mini_batch_size)
                all_valid_preds_0, all_valid_labels_0, all_valid_label_conf_0                         = validate(fpln, validation_set, batch_size=mini_batch_size)
                #calculate_diff_class() no need to calculate diff since it's 0
                #valid_losses[0] = all_valid_losses_0
                valid_preds[0] = all_valid_preds_0
                valid_labels[0] = all_valid_labels_0
                valid_label_conf[0] = all_valid_label_conf_0
                
            
            INNER_tic = time.time()
            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for adapt_idx, (text, labels) in enumerate(support_set):
                
                labels = torch.tensor(labels).to(device)
                input_dict = learner.rln.encode_text(text)
                _repr = learner.rln(input_dict)
                output = fpln(_repr)
                loss = learner.loss_fn(output, labels)
                diffopt.step(loss)
                pred = models.utils.make_prediction(output.detach())
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(labels.tolist())
                
                
                # Test validation_set DURING the update (update=i)
                # Every 5
                #print("ADAPT IDX ", adapt_idx)
                #print(text)
                #print(pred.tolist())
                #if(adapt_idx + 1 % 5 == 0):
                if(True):
                    with torch.no_grad():
                        all_valid_preds_n, all_valid_labels_n, all_valid_label_conf_n =                                 validate(fpln, validation_set, batch_size=mini_batch_size)
                        class_diff_dict = calculate_diff_class(all_valid_labels_0, all_valid_label_conf_0, all_valid_label_conf_n)
                        #valid_losses[adapt_idx+1] = all_valid_losses_n
                        valid_preds[adapt_idx+1] = all_valid_preds_n
                        valid_labels[adapt_idx+1] = all_valid_labels_n
                        valid_label_conf[adapt_idx+1] = all_valid_label_conf_n
                        valid_class_diff[adapt_idx+1] = class_diff_dict
                
                
            INNER_toc = time.time() - INNER_tic
            all_adaptation_time.append(INNER_toc)
            print("Adapt Time: "+ str(INNER_toc//60) +" minutes" )

            acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

            print('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                        'recall = {:.4f}, F1 score = {:.4f}'.format(np.mean(support_loss), acc, prec, rec, f1))

            # Query set is now here!
            query_labels = torch.tensor(query_labels).to(device)
            query_input_dict = learner.rln.encode_text(query_text)
            with torch.no_grad():
                query_repr = learner.rln(query_input_dict)
                query_output = fpln(query_repr) # Output has size of torch.Size([16, 33]) [BATCH, CLASSES]
                query_loss = learner.loss_fn(query_output, query_labels)
            query_loss = query_loss.item()
            # print(output.detach().size())
            # output.detach().max(-1) max on each Batch, which will return [0] max, [1] indices
            query_output_softmax = F.softmax(query_output, -1)
            query_label_conf = query_output_softmax[np.arange(len(query_output_softmax)), query_labels] # Select labels in the softmax of 33 classes

            query_pred = models.utils.make_prediction(query_output.detach())
            query_acc, query_prec, query_rec, query_f1 = models.utils.calculate_metrics(query_pred.tolist(), query_labels.tolist())
            
            print('Query set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                'recall = {:.4f}, F1 score = {:.4f}'.format(np.mean(query_loss), query_acc, query_prec, query_rec, query_f1))

            all_losses.append(query_loss)
            all_predictions.extend(query_pred.tolist())
            all_labels.extend(query_labels.tolist())
            all_label_conf.extend(query_label_conf.tolist())

    acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
    print('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))
    return acc, prec, rec, f1, all_predictions, all_labels, all_label_conf, all_adaptation_time,              valid_preds, valid_labels, valid_label_conf, valid_class_diff


# In[15]:


(4+1)%5==0


# In[16]:


tic = time.time()
print('----------Testing on test set starts here----------')

accuracies, precisions, recalls, f1s = [], [], [], []
all_adapt_time = []
# Data for Visualization: [data_idx, label, label_conf, pred]
data_for_visual = []
# Data for Validation: [data_idx, update, label, label_conf, pred]
data_for_valid = []
#all_class_diff = {} # combine all the class diff

for test_dataset in test_datasets:
    print('Testing on {}'.format(test_dataset.__class__.__name__))
    test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                      collate_fn=datasets.utils.batch_encode)
    acc, prec, rec, f1, all_pred, all_label, all_label_conf, all_adaptation_time,         valid_preds, valid_labels, valid_label_conf, valid_class_diff         = evaluate(dataloader=test_dataloader, updates=updates, mini_batch_size=mini_batch_size, 
                    dataname=test_dataset.__class__.__name__)
    
    data_ids = [test_dataset.__class__.__name__ + str(i) for i in range(len(all_label))]
    data_for_visual.extend(list(zip(data_ids, all_label, all_label_conf, all_pred)))
    all_adapt_time.extend(all_adaptation_time)

    data_valid_ids = [test_dataset.__class__.__name__ + str(i) for i in range(len(valid_labels[0]))]
    
    for adapt_i in valid_labels.keys():
        data_for_valid.extend(list(zip(data_valid_ids, [adapt_i]*len(valid_labels[0]), valid_labels[adapt_i], valid_label_conf[adapt_i], valid_preds[adapt_i])))
    
    print(len(data_valid_ids))
    print(len(valid_labels[0]))
    print(len(valid_preds[0]))
    print(len(valid_label_conf[0]))
    
    print(valid_class_diff.keys())
    print(valid_class_diff[5].keys())
    
    # Valid Label Conf Pickle Dump
    _model_path0 = os.path.splitext(model_path)[0]
    valid_label_conf_filename = _model_path0 + "_update"+ str(updates) +"_valid_label_conf_sr_ta2_" + test_dataset.__class__.__name__ + ".pickle"
    with open(valid_label_conf_filename, "wb") as outfile:
        pickle.dump(valid_label_conf, outfile)
    print(f"Done writing Pickle File at {valid_label_conf_filename}")
    
    # Valid Class Diff Pickle dump
    _model_path0 = os.path.splitext(model_path)[0]
    valid_class_diff_filename = _model_path0 + "_update"+ str(updates) +"_valid_class_diff_sr_ta2_" + test_dataset.__class__.__name__ + ".pickle"
    with open(valid_class_diff_filename, "wb") as outfile:
        pickle.dump(valid_class_diff, outfile)
    print(f"Done writing Pickle File at {valid_class_diff_filename}")
    
    
    # Combine all_class_diff
    #for update_n, class_diff in valid_class_diff.items():
    #    all_class_diff[update_n] = all_class_diff.get(update_n, {})
    #    for class_idx, data_list in class_diff.items():
    #        all_class_diff[update_n][class_idx] = all_class_diff[update_n].get(class_idx, []) + data_list
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)


print()
print("COPY PASTA - not really but ok")
for row in accuracies:
    print(row)
print()
print('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
            'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)))

toc = time.time() - tic
print(f"Total Time used: {toc//60} minutes")


# # Stats
# ```
# Done writing Pickle File at /data/model_runs/original_oml/aOML-order1-inlr010-2022-08-30-sr-query/OML-order1-id4-2022-08-30_05-21-18.854228_update10_valid_results_sr_ta_YelpDataset.pickle
# Testing on AGNewsDataset
# Query ID 0/475
# Adapt Time: 18.0 minutes
# ```

# In[ ]:


# _model_path0 = os.path.splitext(model_path)[0]
# csv_filename = _model_path0 + "_update"+ str(updates) +"_results_sr_ta.csv" # for selective replay
# with open(csv_filename, 'w') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["data_idx", "label", "label_conf", "pred"])
#     csv_writer.writerows(data_for_visual)
# print(f"Done writing CSV File at {csv_filename}")


# In[ ]:


# Log Time for Inference
_model_path0 = os.path.splitext(model_path)[0]
time_txt_filename = _model_path0 + "_update"+ str(updates) +"_time_inference_sr_ta2_es.csv" 
with open(time_txt_filename, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_id", "time"])
    csv_writer.writerow(["Total Time", f"{toc//60} minutes"])
    csv_writer.writerow(["mean Adapt Time", f"{np.mean(all_adapt_time)} s"])
print(f"Done writing Time CSV File at {time_txt_filename}")


# In[ ]:


# Log Valid Set
_model_path0 = os.path.splitext(model_path)[0]
valid_csv_filename = _model_path0 + "_update"+ str(updates) +"_valid_results_sr_ta2_es.csv" # for selective replay
with open(valid_csv_filename, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Data for Validation: [data_idx,label, label_conf, pred]
    csv_writer.writerow(["data_idx","adapt_step", "label", "label_conf", "pred"])
    csv_writer.writerows(data_for_valid)
print(f"Done writing CSV File at {valid_csv_filename}")

