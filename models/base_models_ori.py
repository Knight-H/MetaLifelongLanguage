import math
import random
from collections import defaultdict

import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import AlbertModel, AlbertTokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import numpy as np


class TransformerClsModel(nn.Module):

    def __init__(self, model_name, n_classes, max_length, device):
        super(TransformerClsModel, self).__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        elif model_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.encoder = RobertaModel.from_pretrained('roberta-base')
        else:
            raise NotImplementedError
        self.linear = nn.Linear(768, n_classes)
        self.to(self.device)

    def encode_text(self, text, add_prefix_space=False):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt',
                                                         add_prefix_space=add_prefix_space)
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs, out_from='full'):
        if out_from == 'full':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            out = self.linear(out)
        elif out_from == 'transformers':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        elif out_from == 'linear':
            out = self.linear(inputs)
        else:
            raise ValueError('Invalid value of argument')
        return out


class TransformerRLN(nn.Module):

    def __init__(self, model_name, max_length, device, adapter=False, adapter_type=None):
        super(TransformerRLN, self).__init__()
        self.max_length = max_length
        self.device = device
        self.adapter = adapter
        self.adapter_type = adapter_type
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        elif model_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.encoder = RobertaModel.from_pretrained('roberta-base')
        else:
            raise NotImplementedError
        # If Adapter, add adapter!!
        if self.adapter:
            self.encoder.add_adapter(self.adapter_type)
            self.encoder.set_active_adapters(self.adapter_type) # Sets the adapter modules to be used by default in every forward pass
            print(self.encoder.adapter_summary())

        self.to(self.device)

    def encode_text(self, text, add_prefix_space=False):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt',
                                                         add_prefix_space=add_prefix_space)
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        # Edit, return_dict=False was added when using adapter-transformers
        # https://stackoverflow.com/questions/65082243/dropout-argument-input-position-1-must-be-tensor-not-str-when-using-bert
        # pooler_output (Tensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token)
        if self.adapter:
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=False)  
        else:
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])  
        return out


class LinearPLN(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super(LinearPLN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.to(device)

    def forward(self, input):
        out = self.linear(input)
        return out

class TwoLinearPLN(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super(TwoLinearPLN, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.linear1.to(device)
        self.linear2.to(device)

    def forward(self, input):
        out = self.linear1(input)
        out = self.linear2(out)
        return out


class TransformerNeuromodulator(nn.Module):

    def __init__(self, model_name, device):
        super(TransformerNeuromodulator, self).__init__()
        self.device = device
        if model_name == 'albert':
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError
        self.encoder.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, 768),
                                    nn.Sigmoid())
        self.to(self.device)

    def forward(self, inputs, out_from='full'):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(out)
        return out


class ReplayMemory:

    def __init__(self, write_prob, tuple_size):
        self.buffer = []
        self.write_prob = write_prob
        self.tuple_size = tuple_size

    def write(self, input_tuple, write_prob=None):
        write_prob = write_prob if write_prob else self.write_prob
        if random.random() < write_prob:
            self.buffer.append(input_tuple)

    def read(self):
        return random.choice(self.buffer)

    def write_batch(self, *elements, write_prob=None):
        element_list = []
        for e in elements:
            if isinstance(e, torch.Tensor):
                element_list.append(e.tolist())
            else:
                element_list.append(e)
        for write_tuple in zip(*element_list):
            self.write(write_tuple, write_prob=write_prob)

    def read_batch(self, batch_size):
        contents = [[] for _ in range(self.tuple_size)]
        for _ in range(batch_size):
            if len(self) <= 0:
                break
            read_tuple = self.read()
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        return tuple(contents)

    def len(self):
        return len(self.buffer)
    
    def __len__(self):
        return len(self.buffer)

    def reset_memory(self):
        self.buffer = []
        
        
def convert_to_task_class_key(task_dict):
    task_class_list = []
    for task_idx, class_list in task_dict.items():
        task_class_list.extend([f"{task_idx}|{class_idx}" for class_idx in class_list])
    return task_class_list
        
class LabelAwareReplayMemory:

    def __init__(self, write_prob, tuple_size, n_classes=33, task_dict = {}, validation_split=0., task_aware=True, limit_n=300, filter_support = True):
        # This tells that the key what will be used in the buffer_dict will be task_idx|class_idx instead
        self.task_aware = task_aware
        # This tells whether to use filter condition on overwrite
        self.filter_support = filter_support
        if(self.task_aware):
            self.task_class_list = convert_to_task_class_key(task_dict)
            self.n_total_class = len(self.task_class_list)
        else:
            self.task_class_list = list(range(n_classes))
            self.n_total_class = n_classes
        
        # [0] expo a-n score [1] #support [2] expo a score [3] expo n score
        self.meta_length = 4
        self.limit_n = limit_n
        # These buffers are Map of class_idx -> Array <sample tuples (text , label)>
        self.buffer_dict = {}
        self.buffer_dict_valid = {} # For validation 
        self.meta_score = np.zeros((self.n_total_class, self.limit_n, self.meta_length))
        self.meta_debug = {} # Map of class_idx -> sample_idx -> array of a-n 
        # Metadata of the validation buffer: a-n Score, #Support , debugging list [ a-n ,... ]
        # Map of class_idx -> np Array< score tuple (a-n moving avg, #support, [ a-n, .... ] temp for error tracking) >
        #      No, since this is something that requires sorting, use np better??
        #      This is np array with shape (33, MAX SAMPLE per class, 2) 
        #      --> [0] expo a-n score [1] #support [2] expo a score [3] expo n score
        #      Note: 2,3 is added later so it's theres
        #self.meta_valid_score = np.zeros((self.n_total_class, 300, self.meta_length))
        #self.meta_valid_debug = {}  # Map of class_idx -> sample_idx -> array of a-n 
        # For task_dict (Information about task: task-aware) Map task_idx -> [c1, c2, c3, c4,...]
        self.task_dict = task_dict
        self.n_classes = n_classes
        self.write_prob = write_prob
        self.validation_split = validation_split
        self.tuple_size = tuple_size
        self.read_index = 0
        
    def write(self, input_tuple, task_id=None, write_prob=None, with_index = False):
        write_prob = write_prob if write_prob else self.write_prob
        item_index = None
        if random.random() < write_prob:
            # Write According to Validation Split
            # Remove this logic. There is no more valid split.
            #buffer_to_write = self.buffer_dict
            #if len(self.buffer_dict.get(input_tuple[1], [])) != 0 and \
            #    len(self.buffer_dict_valid.get(input_tuple[1], [])) / len(self.buffer_dict.get(input_tuple[1], [])) < validation_split:
            #    buffer_to_write = self.buffer_dict_valid
            
            key = input_tuple[1] if not self.task_aware else f"{task_id}|{input_tuple[1]}"
            self.buffer_dict[key] = self.buffer_dict.get(key, [])
            # Check if the buffer is full before writing
            is_buffer_full = len(self.buffer_dict[key]) >= self.limit_n 
            if not is_buffer_full:
                self.buffer_dict[key].append([input_tuple[0], *input_tuple[2:]]) # Add everything except [1]
                # print(f"THIS IS len of buffer_dict[{key}] or meta_score id {self.task_class_list.index(key)} : {len(self.buffer_dict[key])} index is {len(self.buffer_dict[key]) - 1} ")
                item_index = len(self.buffer_dict[key]) - 1
                return item_index
            else:
                # If buffer is full and need to write, repalce the one with support >= 1 and is lowest score (other than top 10)
                arr_id = key if not self.task_aware else self.task_class_list.index(key)
                
                all_scores = self.meta_score[arr_id,...,0]
                all_support = self.meta_score[arr_id,...,1]
                sort_indexes = np.argsort(all_scores) # Sort Ascending
                sorted_support = all_support[sort_indexes]
                
                # [A.1] For training with random score (no sort score) / OML where most supports are only updated once!!! (1 support)
                filter_condition = (sorted_support >= 1) if self.filter_support else (sorted_support >= 0)
                #filter_condition = sorted_support >= 0
                sorted_index_filtered = sort_indexes[filter_condition][::-1][10:]
                if(len(sorted_index_filtered) > 10):
                    sorted_index_filtered = sorted_index_filtered[10:] # Get top 10 out too. But if have very few, no need.
                item_index = random.choice(sorted_index_filtered) 
                
                # Replace the index
                self.buffer_dict[key][item_index] = [input_tuple[0], *input_tuple[2:]]
                self.meta_score[arr_id, item_index, ...] = 0
                return item_index
        return item_index
                
            
    def write_batch(self, *elements, task_id=None, write_prob=None, with_index = False):
        assert self.task_aware and task_id != None, "Need task_id if it is task_aware"
        element_list = []
        item_indexes = []
        for e in elements:
            if isinstance(e, torch.Tensor):
                element_list.append(e.tolist())
            else:
                element_list.append(e)
        for write_tuple in zip(*element_list):
            item_index = self.write(write_tuple, task_id=task_id, write_prob=write_prob, with_index=with_index)
            item_indexes.append(item_index)
        return item_indexes
        

    # Same as read, but new item_index returned too
    # Sort Score can only use with meta_score (if it is processed)
    # 1. Filter all with a-n > 0 (also implies support > 0)
    # 2. Get 10th percentile of the remaining items
    def read(self, key, with_index = False, sort_score=False, sort_asc=False, no_support_priority=False):
        if sort_score:
            arr_id = key if not self.task_aware else self.task_class_list.index(key)
            
            all_scores = self.meta_score[arr_id,...,0]
            all_support = self.meta_score[arr_id,...,1]
            all_a_scores = self.meta_score[arr_id,...,2]
            sort_indexes = np.argsort(all_scores) # Sort Ascending
            sorted_array = all_scores[sort_indexes]       # Get the sorted array to get percentile
            sorted_support = all_support[sort_indexes]
            sorted_a = all_a_scores[sort_indexes]
            
            #zero_idx = np.searchsorted(sorted_array, 0, side='right') # binary search to the first from right
            #p = np.percentile(sorted_array[zero_idx:], 90)
            #p_idx = np.searchsorted(sorted_array, p) # binary search to the first from left
            
            # Use TopK instead!!!
            
            #item_index = random.choice(sort_indexes[p_idx:])
            sorter = 1 if sort_asc else -1
            filter_condition = (sorted_support >= 3) & (sorted_a >= 0.5)
            item_index = random.choice(sort_indexes[filter_condition][::sorter][:20]) # Filter all with support <3, and descending , top 20
            class_id = key if not self.task_aware else int(key.split("|")[-1])
        elif no_support_priority:
            arr_id = key if not self.task_aware else self.task_class_list.index(key)
            
            all_scores = self.meta_score[arr_id,...,0]
            all_support = self.meta_score[arr_id,...,1]
            all_a_scores = self.meta_score[arr_id,...,2]
            sort_indexes = np.argsort(all_support) # Sort Ascending
            sorted_array = all_scores[sort_indexes]       # Get the sorted array 
            sorted_support = all_support[sort_indexes]
            sorted_a = all_a_scores[sort_indexes]
            
            # Use TopK instead!!!
            # Filter on existing data only!
            n_data = len(self.buffer_dict[key])
            filter_condition = (sort_indexes < n_data)
            item_index = random.choice(sort_indexes[filter_condition][:20]) 
            class_id = key if not self.task_aware else int(key.split("|")[-1])
        else:
            # Normal case:
            item_index = random.choice(range(len(self.buffer_dict[key])))
            class_id = key if not self.task_aware else int(key.split("|")[-1])
        
#         print(f"arr_id {arr_id}")
#         print(f"key {key}")
#         print(f"item_index {item_index}")
#         print(f"self.buffer_dict[key] {len(self.buffer_dict[key])} {self.buffer_dict[key]}")
        output = [self.buffer_dict[key][item_index][0], class_id, *self.buffer_dict[key][item_index][1:]]
        if with_index:
            output += [item_index]
        return output
    

    def read_batch(self, batch_size, with_index=False):
        tuple_size = self.tuple_size
        if with_index: # for the index too!
            tuple_size += 1
        contents = [[] for _ in range(tuple_size)]

        # Read batch always random!
        iid = random.choice(range(self.n_total_class))
        self.read_index = self.task_class_list[iid]

        for _ in range(batch_size):
            # If buffer doesn't have, move to next 
            while not self.read_index in self.buffer_dict:
                iid = (iid + 1)%self.n_total_class
                self.read_index = self.task_class_list[iid]
            read_tuple = self.read(self.read_index, with_index=with_index)
            # [Round Robin!]
            iid = (iid + 1)%self.n_total_class 
            self.read_index = self.task_class_list[iid]
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        return tuple(contents)
    
    # with_index = also return the index 
    # sort_score = sort the score by sort_asc before returning
    # random_class = don't start with task_class_idx 0 (useful when minibatch = 1, and it will always return task_class_idx=0)
    # random_task = also random the example's task in each batch from the task pool
    # no_support_priority: boolean = return where there is no support first (something like sort support)
    # with_number_samples: boolean = also return the total number of samples inside that task (used for 
    def read_batch_task(self, batch_size, task_idx, with_index=False, sort_score=False, sort_asc=False, random_class = False, \
                       random_task = False, no_support_priority=False, with_number_samples=False):
        tuple_size = self.tuple_size
        if with_index: # for the index too!
            tuple_size += 1
        
        if random_task and self.task_aware: # if we need to random_task in each sample of the batch, we need to set up the task pool (task with examples)
            task_pool = list(set([int(key.split("|")[0]) for key in self.buffer_dict if len(self.buffer_dict[key]) > 0]))
            #print("task_pool")
            #print(self.task_dict.keys())
            #print(task_pool)
        # If task_idx is None, means we initially random task_idx too!
        if task_idx is None:
            task_idx = random.choice(task_pool)
        
        is_out_tries = False

        contents = [[] for _ in range(tuple_size)]
        # Set read_index to start with the first class of the task_idx
        task_class_idx = 0 if not random_class else random.choice(range(len(self.task_dict[task_idx])))
        class_idx = self.task_dict[task_idx][task_class_idx] 
        self.read_index = class_idx if not self.task_aware else f"{task_idx}|{class_idx}" # class to read in that task / if task aware -> task_id|class_id
        
        for _ in range(batch_size):
            total_retries = 30
            # If buffer doesn't have, move to next
            while not self.read_index in self.buffer_dict:
                task_class_idx = (task_class_idx + 1)%len(self.task_dict[task_idx]) # go on to the next class in the task
                class_idx = self.task_dict[task_idx][task_class_idx]
                self.read_index = class_idx if not self.task_aware else f"{task_idx}|{class_idx}"
                total_retries -= 1
                if total_retries < 0:
                    is_out_tries = True
                    break
            if is_out_tries:
                break
            
            read_tuple = self.read(self.read_index, with_index=with_index, sort_score=sort_score, sort_asc=sort_asc, no_support_priority=no_support_priority)
            if random_task: # If random_task, also need to random task_idx
                task_idx = random.choice(task_pool)
            task_class_idx = (task_class_idx + 1)%len(self.task_dict[task_idx]) # go on to the next class in the task
            class_idx = self.task_dict[task_idx][task_class_idx] 
            self.read_index = class_idx if not self.task_aware else f"{task_idx}|{class_idx}"
            
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        # Also return the number of samples in that task for this arg
        if with_number_samples:
            contents += [self.get_task_number_samples(task_idx=task_idx)]
        return tuple(contents)
    
    def get_task_number_samples(self,task_idx):
        if self.task_aware:
            return sum([len(self.buffer_dict.get(f"{task_idx}|{class_idx}", [])) for class_idx in self.task_dict[task_idx]])
        else: 
            return 0
    def reset_memory(self):
        self.buffer_dict = {}
    def reset_meta(self, limit_n = None):
        self.limit_n = limit_n if limit_n else self.limit_n
        self.meta_score = np.zeros((self.n_total_class, self.limit_n, self.meta_length))
        self.meta_debug = {} # Map of class_idx -> sample_idx -> array of a-n 
    
    # For hacking the early stopping to work!!
    # Split the buffer_dict (train) to buffer_dict_valid (validation)
    def split_train_validation(self, limit_n = 300):
        self.buffer_dict_valid = {} # Reset the buffer_dict_valid
        for class_idx, buffer_list in self.buffer_dict.items():
            self.buffer_dict_valid[class_idx] = []
            for i in range(limit_n):
                pop_data = self.buffer_dict[class_idx].pop(random.randint(0,len(self.buffer_dict[class_idx])-1))
                self.buffer_dict_valid[class_idx].append(pop_data)
        return 
    # hack trimming the buffer dict for easier sorting.
    def trim_buffer_dict(self, limit_n = 1000):
        temp_buffer_dict = {} 
        for key, buffer_list in self.buffer_dict.items():
            temp_buffer_dict[key] = []
            for i in range(limit_n):
                pop_data = self.buffer_dict[key].pop(random.randint(0,len(self.buffer_dict[key])-1))
                temp_buffer_dict[key].append(pop_data)
        self.buffer_dict = temp_buffer_dict
        return 
    
    # Update the Meta Data: Expo a-n, support, a,n debug
    def update_meta(self, class_indexes, sample_indexes, non_adapt_arr, adapt_arr, conf_diff, write_debug = True, discount= 0.3, task_id=None):
        assert self.task_aware and task_id != None, "Need task_id if it is task_aware" 
        assert len(class_indexes)== len(sample_indexes), "class Idx and sample Idx should be equal"
        assert len(sample_indexes) == len(conf_diff), "Sample and diff should be equal"
        assert len(non_adapt_arr) == len(conf_diff), "n and n-a should be equal"
        assert len(non_adapt_arr) == len(adapt_arr), "n and a should be equal"

        # print("Full sample Indexes ", sample_indexes)

        # Make sure that it is numpy
        non_adapt_arr = np.array(non_adapt_arr)
        adapt_arr = np.array(adapt_arr)
        conf_diff = np.array(conf_diff)  # Actually this can be calculated here if u want too!!      
        class_indexes = np.array(class_indexes)
        sample_indexes = np.array(sample_indexes)

        # Sample Indexes may contain None If it is not reverse support (Not everything is written because it's new). ie. D_S <-- DS instead of Mem
        # Filter everything to ensure that has number!
        filter_none = sample_indexes != None
        class_indexes = class_indexes[filter_none]
        conf_diff = conf_diff[filter_none]
        adapt_arr = adapt_arr[filter_none]
        non_adapt_arr = non_adapt_arr[filter_none]
        sample_indexes = sample_indexes[filter_none]
        
        keys = class_indexes if not self.task_aware else [self.task_class_list.index(f"{task_id}|{class_id}") for class_id in class_indexes]
        sample_indexes = sample_indexes.tolist()  # Can actually just make both keys and sample indexes int np array, but lazy to convert. Else you cant use it to index!!!

        # Initialize ones that have 0 support
        filter_zero_support = self.meta_score[keys, sample_indexes, 1] <= 0
        self.meta_score[keys, sample_indexes, 0][filter_zero_support] = conf_diff[filter_zero_support]
        self.meta_score[keys, sample_indexes, 2][filter_zero_support] = adapt_arr[filter_zero_support]
        self.meta_score[keys, sample_indexes, 3][filter_zero_support] = non_adapt_arr[filter_zero_support]
        
        # Actually can do it with numpy array .. but then same index will not work ?
        # fok it. just do it.? shouldn't have same index anyways?!?
        self.meta_score[keys, sample_indexes, 0] *= 1-discount
        self.meta_score[keys, sample_indexes, 0] += discount * conf_diff
        self.meta_score[keys, sample_indexes, 2] *= 1-discount
        self.meta_score[keys, sample_indexes, 2] += discount * adapt_arr
        self.meta_score[keys, sample_indexes, 3] *= 1-discount
        self.meta_score[keys, sample_indexes, 3] += discount * non_adapt_arr
        self.meta_score[keys, sample_indexes, 1] += 1 # Support
        # print("KEYS ", keys)
        # print("sample_indexes", sample_indexes)
        # print("UPDATE META SUPPORT ", self.meta_score[keys, ..., 1])
        
        if write_debug:
            for i, key in enumerate(keys):
                #print(f"meta_debug keys : {self.meta_debug.keys()}")
                self.meta_debug[key] = self.meta_debug.get(key, {})
                #print(f"meta_debug keys after : {self.meta_debug.keys()}")
                self.meta_debug[key][sample_indexes[i]] = self.meta_debug[key].get(sample_indexes[i], []) + [conf_diff[i]]
                #print(f"meta_debug class keys after : {self.meta_debug[class_idx].keys()}")
                #print(f"meta_debug i : {i} , {class_idx}")
                #print(f"meta_debug conf diff : {conf_diff[i]}")
            #raise Exception("BREAKPOINT")
            
    
    # Read the whole validation set of a certain task
    def read_batch_validation_task(self, task_idx):
        all_batch_data = []
        all_batch_classes = []
        for class_idx in task_dict[task_idx]:
            all_batch_data.extend(self.buffer_dict_valid[class_idx])
            all_batch_classes.extend([class_idx for _ in range(len(self.buffer_dict_valid[class_idx]))])
        return (all_batch_data, all_batch_classes)
    
    # Read the whole validation set 
    def read_batch_validation(self):
        all_batch_data = []
        all_batch_classes = []
        for class_idx in range(self.n_classes):
            all_batch_data.extend(self.buffer_dict_valid[class_idx])
            all_batch_classes.extend([class_idx for _ in range(len(self.buffer_dict_valid[class_idx]))])
        return (all_batch_data, all_batch_classes)
    