import os
from torch.utils.data import Dataset, DataLoader, Sampler
import json, csv
from collections import OrderedDict
from multiprocessing import Pool
import torch
import torch.nn.functional as F
import re
import logging



LEN_FACTOR = 1.163
FILL_VAL = -1

def getTaskDict(data_dir):
    TASK_DICT =  {
        "movie": {
                   "train":os.path.join(data_dir,"movie_train.json"),
                   "eval":os.path.join(data_dir,"movie_dev.json"),
                   "test":os.path.join(data_dir,"movie_test.json"),
        },
        "boolq": {
                   "train":os.path.join(data_dir,"boolq_train.json"),
                   "eval":os.path.join(data_dir,"boolq_dev.json"),
                   "test":os.path.join(data_dir,"boolq_test.json"),
        },
        "scifact": {
                   "train":os.path.join(data_dir,"scifact_train.json"),
                   "eval":os.path.join(data_dir,"scifact_dev.json"),
                   "test":os.path.join(data_dir,"scifact_test.json"),
        },
        "sst": {
                   "train":os.path.join(data_dir,"sst_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"sst_to_squad-dev-v2.0.json"),
                   "test":os.path.join(data_dir,"sst_to_squad-test-v2.0.json"),
        },
        "srl": {
                   "train":os.path.join(data_dir,"srl_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"srl_to_squad-dev-v2.0.json"),
                   "test":os.path.join(data_dir,"srl_to_squad-test-v2.0.json"),
        },
        "woz.en": {
                   "train":os.path.join(data_dir,"woz.en_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"woz.en_to_squad-dev-v2.0.json"),
                   "test":os.path.join(data_dir,"woz.en_to_squad-test-v2.0.json"),
        },
        "ag": {
                   "train":os.path.join(data_dir,"ag_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"ag_to_squad-test-v2.0.json"),
                   "test":os.path.join(data_dir,"ag_to_squad-test-v2.0.json"),
        },
        "dbpedia": {
                   "train":os.path.join(data_dir,"dbpedia_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"dbpedia_to_squad-test-v2.0.json"),
                   "test":os.path.join(data_dir,"dbpedia_to_squad-test-v2.0.json"),
        },
        "yahoo": {
                   "train":os.path.join(data_dir,"yahoo_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"yahoo_to_squad-test-v2.0.json"),
                   "test":os.path.join(data_dir,"yahoo_to_squad-test-v2.0.json"),
        },
        "amazon": {
                   "train":os.path.join(data_dir,"amazon_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"amazon_to_squad-test-v2.0.json"),
                   "test":os.path.join(data_dir,"amazon_to_squad-test-v2.0.json"),
        },
        "yelp": {
                   "train":os.path.join(data_dir,"yelp_to_squad-train-v2.0.json"),
                   "eval":os.path.join(data_dir,"yelp_to_squad-test-v2.0.json"),
                   "test":os.path.join(data_dir,"yelp_to_squad-test-v2.0.json"),
        },
        "ag10k": {
                   "train":os.path.join(data_dir,"ag_to_squad-train-v2.0-10k.json"),
                   "eval":os.path.join(data_dir,"ag_to_squad-test-v2.0-10k.json"),
                   "test":os.path.join(data_dir,"ag_to_squad-test-v2.0-10k.json"),
        },
        "dbpedia10k": {
                   "train":os.path.join(data_dir,"dbpedia_to_squad-train-v2.0-10k.json"),
                   "eval":os.path.join(data_dir,"dbpedia_to_squad-test-v2.0-10k.json"),
                   "test":os.path.join(data_dir,"dbpedia_to_squad-test-v2.0-10k.json"),
        },
        "yahoo10k": {
                   "train":os.path.join(data_dir,"yahoo_to_squad-train-v2.0-10k.json"),
                   "eval":os.path.join(data_dir,"yahoo_to_squad-test-v2.0-10k.json"),
                   "test":os.path.join(data_dir,"yahoo_to_squad-test-v2.0-10k.json"),
        },
        "amazon10k": {
                   "train":os.path.join(data_dir,"amazon_to_squad-train-v2.0-10k.json"),
                   "eval":os.path.join(data_dir,"amazon_to_squad-test-v2.0-10k.json"),
                   "test":os.path.join(data_dir,"amazon_to_squad-test-v2.0-10k.json"),
        },
        "yelp10k": {
                   "train":os.path.join(data_dir,"yelp_to_squad-train-v2.0-10k.json"),
                   "eval":os.path.join(data_dir,"yelp_to_squad-test-v2.0-10k.json"),
                   "test":os.path.join(data_dir,"yelp_to_squad-test-v2.0-10k.json"),
        }
    }
    return TASK_DICT


# ```python
# return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx
#            0                1               2           3                   4       5              6          7
# # 0 cq_example is context+question+__ans__. ie. [7110, 25, 734, 6036, 11886, 467, 284, 257, 4928, 2151]
# # 1 len(cq_example) is the length ie. 901
# # 2 cqa_example is context+question+__ans__+answer ie. [7110, 25, 734, 6036, 11886, 467, 284, 257, 4928, 2151]
# # 3 len(cqa_example) is the length ie. 903
# # 4 Y_example is FILL_VALUE+answer only. ie. [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
# # 5 gen_X_example is __gen__+context+question+__ans__+answer ie. [50260, 7110, 25, 734, 6036, 11886, 467, 284, 257, 4928]
# # 6 gen_Y_example is context+question+__ans__+answer ie. [7110, 25, 734, 6036, 11886, 467, 284, 257, 4928, 2151]
# # 7 idx is id (supposed to be uuid? but i don't see it) ie. 0
# ```
# train_qadata = QADataset(TASK_DICT[_tasks[0]]["train"], "train", SPECIAL_TOKEN_IDS[_tasks[0]], train_extra_data)
# max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
# train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
class QADataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, special_token_ids, tokenizer, extra_data=[], **kwargs):
        self.data_type = data_type
        self.gen_token = gen_token
        self.ans_token = special_token_ids["ans_token"]
        self.eos_token = special_token_ids["eos_token"]
        self.pad_token = special_token_ids["pad_token"]
        self.tokenizer = tokenizer
        self.data_dir = kwargs.get('data_dir')
        self.max_len = kwargs.get('max_length')
        self.tokenize_n_cpu = kwargs.get('tokenize_n_cpu')

        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        
        # actual loading of data using json.load()
        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d
        
        # tokenize data ???
        self.data = []
        self.max_a_len = 0
        if len(data_paths)==1 and data_paths[0] is not None and ('wiki' in data_paths[0] or 'woz' in data_paths[0]):
            if 'wiki' in data_paths[0]:
                answers_file = "wikisql_answers.json" 
            elif 'woz' in data_paths[0]:
                answers_file = "woz.en_answers.json" 
            with open(os.path.join(self.data_dir,answers_file),"r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            self.data_tokenization(data)

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > self.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:self.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
        cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
        gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx

    def parallel_tokenization(self, d):
        examples = []
        
        # Tokenize context
        context = self.tokenizer.encode(d["context"], truncation = True)
        max_a_len = 0
        for qa in d["qas"]:
            # Tokenize question
            question = self.tokenizer.encode(qa["question"], truncation = True)
            
            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})
            
            # Tokenize Answer
            answer = []
            for i, raw_answer in enumerate(raw_answers):
                answer.extend(self.tokenizer.encode(raw_answer["text"], truncation = True))
                if i != len(raw_answers) - 1:
                    answer.append(self.pad_token)
            # Max answer length
            max_a_len = max(max_a_len, len(answer))

            # Append examples with __gentoken__ , ctx, question, answer, question ID
            examples.append(self.parse_example(self.gen_token, context, question, answer, qa.get("id", 0)))
        return examples, max_a_len

    def data_tokenization(self, data):
        with Pool(self.tokenize_n_cpu) as pool:
            data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: len(x[0]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x[-1])

    def get_indices(self):
        return [d[-1] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    
def get_gen_token(task):
    return '__' + task + '__'


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size, **kwargs):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = kwargs.get('train_batch_size')
        else:
            self.batch_size = kwargs.get('test_batch_size')
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size

    def __iter__(self):
        if self.data_type == "test":
            indices = range(self.n_samples)
        else:
            indices = np.random.permutation(self.n_samples)
        max_len, cnt, st = 0, 0, 0
        batch = []
        for ed, idx in enumerate(indices):
            ln = len(self.dataset[idx][2])
            if max(max_len, ln)**LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt]:
                st = ed
                cnt += 1
                max_len = 0
                if cnt == kwargs.get('n_gpus'):
                    yield batch
                    cnt = 0
                    batch = []
            max_len = max(max_len, ln)
            batch.append(idx)
            if len(batch) == self.max_batch_size and self.data_type == "train":
                yield batch
                cnt, max_len, st = 0, 0, ed
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError



def create_dataloader(dataset, data_type, special_token_ids, max_batch_size=1000000000, **kwargs):
    n_gpus = kwargs.get('n_gpus')
    if data_type == "train":
        batch_size = kwargs.get('train_batch_size')
    else:
        batch_size = kwargs.get('test_batch_size')

    if isinstance(batch_size, list):
        collate_fn=lambda x,bs=batch_size: dynamic_collate_fn(x, bs, special_token_ids)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size, **kwargs)
    else:
        collate_fn=lambda x: varlen_collate_fn(x, special_token_ids, n_gpus)
#         shuffle = not (data_type != "train" or args.debug)
        shuffle = False
        batch_sampler = None

    dataloader =  DataLoader(dataset, num_workers=4,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             batch_sampler=batch_sampler)
    return dataloader


def dynamic_collate_fn(data, batch_size, special_token_ids):

    def local_collate():
        null_counter = 0
        _cqs, _len_cqs, _cqas, _len_cqas, _Ys, _gen_Xs, _gen_Ys = [], [], [], [], [], [], []
        Y_max_len = max(len(data[j][4]) for j in range(st, ed))
        cq_max_len = max(len(data[j][0]) for j in range(st, ed))
        for j in range(st, ed):
            if None in data[j] or [] in data[j]:
                null_counter+=1
                logger.warning('null example in collate_fn, count: {}'.format(null_counter))
                continue

            pad_len = cqa_max_len - len(data[j][2])

            _cqs.append(pad_to_max_len(data[j][0], cq_max_len-len(data[j][0]), special_token_ids["pad_token"]))
            _len_cqs.append(data[j][1])
            _cqas.append(pad_to_max_len(data[j][2], pad_len, special_token_ids["pad_token"]))
            _len_cqas.append(data[j][3])
            _Ys.append(pad_to_max_len(data[j][4], Y_max_len - len(data[j][4]), FILL_VAL))
            _gen_Xs.append(pad_to_max_len(data[j][5], pad_len, special_token_ids["pad_token"]))
            _gen_Ys.append(pad_to_max_len(data[j][6], pad_len, FILL_VAL))

        cqs.append(torch.tensor(_cqs))
        len_cqs.append(torch.tensor(_len_cqs))
        cqas.append(torch.tensor(_cqas))
        len_cqas.append(torch.tensor(_len_cqas))
        Ys.append(torch.tensor(_Ys))
        gen_Xs.append(torch.tensor(_gen_Xs))
        gen_Ys.append(torch.tensor(_gen_Ys))

    cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys = [], [], [], [], [], [], []
    cqa_max_len, cnt, st = 0, 0, 0
    for ed, datum in enumerate(data):
        ln = len(datum[2]) # use cqas to calibrate
        if max(cqa_max_len, ln)**LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
            local_collate()
            cnt += 1
            cqa_max_len = 0
            st = ed
        cqa_max_len = max(cqa_max_len, ln)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys


def varlen_collate_fn(data, special_token_ids, n_gpus=1):
    batch_size = (len(data) + n_gpus - 1) // n_gpus
    cqs = torch.tensor(pad_all_to_max_len([datum[0] for datum in data], special_token_ids["pad_token"])).split(batch_size)
    len_cqs = torch.tensor([datum[1] for datum in data]).split(batch_size)
    cqas = torch.tensor(pad_all_to_max_len([datum[2] for datum in data], special_token_ids["pad_token"])).split(batch_size)
    len_cqas = torch.tensor([datum[3] for datum in data]).split(batch_size)
    Ys = torch.tensor(pad_all_to_max_len([datum[4] for datum in data], FILL_VAL)).split(batch_size)    
    gen_Xs = torch.tensor(pad_all_to_max_len([datum[5] for datum in data], special_token_ids["pad_token"])).split(batch_size)
    gen_Ys = torch.tensor(pad_all_to_max_len([datum[6] for datum in data], FILL_VAL)).split(batch_size)
    return list(cqs), list(len_cqs), list(cqas), list(len_cqas), list(Ys), list(gen_Xs), list(gen_Ys)

def pad_to_max_len(l, pad_len, val):
    return l + [val] * pad_len
def pad_all_to_max_len(ls, val):
    max_len = max(len(l) for l in ls)
    return [pad_to_max_len(l, max_len-len(l), val) for l in ls]




def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits



def logits_to_tokens(next_logits, **kwargs):
    top_k_qa = kwargs.get('top_k_qa', 0)
    top_p_qa = kwargs.get('top_p_qa', 0.0)
    
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=top_k_qa, top_p=top_p_qa)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens


## SAMPLE SEQUENCe
def remove_id(idx, need_process, all_pasts, n_layer):
    assert idx in need_process
    del need_process[idx]
    for layer_id in range(n_layer):
        all_pasts[layer_id][idx] = 0
def sample_sequence(rln, pln, need_process, qa_results, all_pasts, max_tot_lens, **kwargs):
    test_batch_size = kwargs.get('test_batch_size')
    
    while len(need_process) > 0:
        first_id = next(iter(need_process))
        shortest_len = len(qa_results[first_id])
#         decode_batch_size = int(args.memory_sizes[0] * MEMORY_FACTOR[args.seq_train_type] // (shortest_len+1)**LEN_FACTOR)
        decode_batch_size = test_batch_size
        it = iter(need_process)
        stop = False
        remove_ids = []
        while not stop:
            batch_ids, input_ids, past = [], [], [[] for _ in range(rln.config.n_layer)]
            while True:
                try:
                    cur_id = next(it)
                    if len(qa_results[cur_id]) > shortest_len:
                        stop = True
                        break
                    batch_ids.append(cur_id)
                    input_ids.append(qa_results[cur_id][-1:])
                    for layer_id in range(MODEL_CONFIG.n_layer):
                        past[layer_id].append(all_pasts[layer_id][cur_id])
                    if len(input_ids) == decode_batch_size:
                        break
                except StopIteration:
                    stop = True
                    break

            n_inputs = len(input_ids)
            if n_inputs == 0:
                break
            input_ids = torch.stack(input_ids)
            for layer_id in range(MODEL_CONFIG.n_layer):
                past[layer_id] = torch.stack(past[layer_id], dim=1)
            all_outputs = model(input_ids=input_ids.cuda(), past=past)

            outputs = all_outputs[0]
            pasts = all_outputs[1]

            next_logits = outputs[..., -1, :] / args.temperature_qa
            next_tokens = logits_to_tokens(next_logits).cpu()

            for i, cur_id in enumerate(batch_ids):
                if next_tokens[i] == SPECIAL_TOKEN_IDS["eos_token"]:
                    remove_ids.append(cur_id)
                else:
                    qa_results[cur_id] = torch.cat((qa_results[cur_id], next_tokens[i]))
                    if len(qa_results[cur_id]) in [max_tot_lens[cur_id], args.max_len]:
                        remove_ids.append(cur_id)
                    else:
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            all_pasts[layer_id][cur_id] = pasts[layer_id][:, i].type(torch.half)
        for idx in remove_ids:
            remove_id(idx, need_process, all_pasts)
