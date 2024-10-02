import logging, time
import math
import os, pickle, csv, json

import higher
import torch
from torch import nn, optim

import numpy as np

from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils import data
from torch.optim import AdamW

# Somehow not explicitly importing utils_knight led to some errors https://stackoverflow.com/questions/1250103/attributeerror-module-object-has-no-attribute, can reload or just import it directly
import datasets.utils
from datasets.utils_knight import QADataset, create_dataloader, FILL_VAL, logits_to_tokens, remove_id
import models.utils
from models.base_models import TransformerRLN, LinearPLN, ReplayMemory, CustomGPT2LMHeadPLN, CustomGPT2RLN
from .metrics import computeEM, get_test_score, compute_metrics


DATA_MAP = {
    0: 'ag',
    1: 'amazon',
    2: 'yelp',
    3: 'dbpedia',
    4: 'yahoo'
}

class OML2:
    
    def __init__(self, device, n_classes, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.model_dir = kwargs.get('model_dir')
        self.run_id = kwargs.get('run_id')
        self.data_dir  = kwargs.get('data_dir')
        self.test  = kwargs.get('test', False)
        self.device = device
        
        self.rln = CustomGPT2RLN.from_pretrained('gpt2', 
                                                 ignore_mismatched_sizes = True,
                                                 max_length=kwargs.get('max_length'),
                                                 token_weight=kwargs.get('token_weight'))
        self.rln.add_special_tokens_init(device)
        self.pln = CustomGPT2LMHeadPLN.from_pretrained('gpt2', 
                                                       config = self.rln.config, 
                                                       input_embeddings = self.rln.get_input_embeddings())
        self.rln.to(device)
        self.pln.to(device)
        
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        
        # Set logger to both the console and the file
        # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
        # https://stackoverflow.com/questions/6614078/logging-setlevel-how-it-works
        self.logger = logging.getLogger('OML2-Log')
        self.logger.setLevel(level=logging.INFO)
        logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fileHandler = logging.FileHandler(f'{self.model_dir}/{self.run_id}.{"test" if self.test else "log"}')
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
    
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)
        
        self.logger.propagate = False

        self.logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        self.logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))


    def save_model(self, model_path):
        checkpoint = {'rln': self.rln.state_dict(),
                      'pln': self.pln.state_dict()}
        torch.save(checkpoint, model_path)

    def load_model(self, model_path, map_location='cuda:0'):
        checkpoint = torch.load(model_path, map_location)
        self.rln.load_state_dict(checkpoint['rln'])
        self.pln.load_state_dict(checkpoint['pln'])

    

    # For Training
    # data_order is the list of the data indexes ie. [2, 0, 3, 1, 4]
    def training(self, data_order, **kwargs):
        self.logger.info("Starting Training code!!!")
        
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        
        n_epochs = kwargs.get('n_epochs')
        
        # Some random thing from lamol dont know if can remove or not
        min_n_steps = kwargs.get('min_n_steps')
        min_batch_size = kwargs.get('min_batch_size')
        
        is10k = kwargs.get('10k')
        
        # Calculate Replays
        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // mini_batch_size
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 1)))
            replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)
        else:
            replay_freq = 0
            replay_steps = 0
        self.logger.info('Replay frequency: {}'.format(replay_freq))
        self.logger.info('Replay steps: {}'.format(replay_steps))
        
        # For every task in the task order
        for data_idx in data_order:
            tic_TASK = time.time()
            
            dataname = DATA_MAP[data_idx]  # This gets the name ag/yahoo/yelp......
            self.logger.info(f"Starting task {dataname}")
            
            ##### Start training on task_id #####
            gen_token = datasets.utils_knight.get_gen_token(dataname)
            self.rln.tokenizer.add_tokens([gen_token])
            self.rln.tokenizer.save_pretrained(self.model_dir)
            self.rln.special_tokens[dataname] = gen_token
            self.rln.special_token_ids[dataname] = self.rln.tokenizer.convert_tokens_to_ids(gen_token)
            self.logger.info('gen token = {} , gen token id = {}'.format(gen_token, self.rln.special_token_ids[dataname]))
            self.rln.config.vocab_size = len(self.rln.tokenizer)
            self.rln.config.to_json_file(os.path.join(self.model_dir,f'{dataname}_model_config.json'))
            if len(self.rln.tokenizer) != self.rln.tokens_weight.shape[0]:
                self.rln.tokens_weight = torch.cat((self.rln.tokens_weight, torch.ones([1]).cuda()))
            # Resize Token Embeddings after special tokens are appended
            self.rln.resize_token_embeddings(len(self.rln.tokenizer))
            self.pln.tie_weights(self.rln.get_input_embeddings()) # tie weights again if needed
            
            ######  Optimizer ######
            # This needs to be located here!! else the PLN weights (in higher) will make error.
            # Because pln is always changing weights through changing rln.
            # Params for Meta adaptation
            inner_params = [p for p in self.pln.parameters() if p.requires_grad]
            self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)
            
            # Moving this here too (not sure if it matters or not but okay
            # Params for Meta Learning
            meta_params = [p for p in self.rln.named_parameters() if p[1].requires_grad] + \
                          [p for p in self.pln.named_parameters() if p[1].requires_grad]  # [0] is name, [1] is Param
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            meta_grouped_params = [
                {'params': [p for n, p in meta_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in meta_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.meta_optimizer = AdamW(meta_grouped_params, lr=self.meta_lr, eps=1e-4) #Copy from LAMOL

            
            # Get Dataset and DataLoader
            # Load the datasets - Moving load datasets here!!!
            tic_DATALOAD = time.time()
            self.logger.info(f"Loading the dataset {dataname}")
            
            
            dataname_key = dataname+"10k" if is10k else dataname # TRAIN ALL 10 k!!
            train_dir    = datasets.utils_knight.getTaskDict(self.data_dir)[dataname_key]["train"] 
            train_qadata = QADataset(train_dir, 
                                     "train", 
                                     self.rln.special_token_ids[dataname], 
                                     self.rln.special_token_ids, 
                                     self.rln.tokenizer,
                                     [], **kwargs)
            max_train_batch_size = max(len(train_qadata) // min_n_steps, min_batch_size)
            train_dataloader = create_dataloader(train_qadata, "train", self.rln.special_token_ids, max_train_batch_size, **kwargs)
            
            toc_DATALOAD = time.time() - tic_DATALOAD
            self.logger.info(f'Finished loading the dataset {dataname} in {toc_DATALOAD/60} mins')
            
            
            # Training loss function - here since tokens weight may die! (change at new tokens)
            train_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=self.rln.tokens_weight)
            
            
            n_steps = 0
            
            episode_id = 0
            episode_loss = []
            
            for epoch in range(n_epochs):
                self.logger.info(f'Starting epoch {epoch}')
                ##### Stream from that dataset's dataloader #####
                iter_dataloader = iter(train_dataloader)

                while True:
                    tic_BATCH = time.time()

                    self.inner_optimizer.zero_grad()
                    support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []

                    # https://github.com/facebookresearch/higher
                    # A context manager for writing differentiable inner loops.
                    # copy_initial_weights -  If this is set to False, the actual module weights will be the initial weights of the patched module. This is useful when doing MAML, for example.
                    # track_higher_grads - Setting this to False allows innerloop_ctx to be used in “test mode”, without potentially tracking higher order gradients. This can be useful when running the training loop at test time, e.g. in k-shot learning experiments, without incurring a significant memory overhead.
                    # Yields
                    # A (fmodule, diffopt) tuple. where fmodule is a “stateless” version of the original module, for which calls to forward take the additional kwarg-only parameter params, which should be a list of torch tensors requiring gradients, ideally provided by this function (see below) or by an update step from one of the optimizers in higher.optim. And diffopt is an initialized DifferentiableOptimizer instance of the right subtype.
                    self.pln.train()
                    with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                              copy_initial_weights=False,
                                              track_higher_grads=False) as (fpln, diffopt):

                        # Inner loop - make support set equal size to the updates
                        # Since OML shows that using SGD update on single batches are better!!!

                        support_set = []
                        task_predictions, task_labels = [], []
                        for _ in range(updates):
                            try: 
                                _, _, support_x, _, support_y, support_gen_x, support_gen_y = next(iter_dataloader)

                                # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
                                support_x = support_x[0] # the size is [1 (batch), seqlen]
                                support_y = support_y[0]

                                support_set.append((support_x, support_y))
                            except StopIteration:
                                self.logger.info('Terminating training as all the data is seen')
                                break

                        n_inputs = sum(_cqa.shape[0] for _cqa in support_x)

                        for support_x, support_y in support_set:
                            support_y = support_y.to(self.device)
                            support_x = support_x.to(self.device)
                            repr_ = self.rln(support_x)[0]                                  # pass through rln (GPT2)
                            output = fpln(repr_)                                          # use functional (linear) pln passthrough

                            qa_logits = output[0]
                            qa_loss = train_loss_fct(qa_logits.transpose(1,2), support_y)
                            loss = qa_loss
                            diffopt.step(loss)
                            support_loss.append(loss.item())

                            # Write this later!!
                            #pred = models.utils.make_prediction(output.detach())
                            #task_predictions.extend(pred.tolist())
                            #task_labels.extend(labels.tolist())
                            self.memory.write_batch(support_x, support_y)



                        #acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

    #                     logger.info('Episode {} support set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
    #                                 'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
    #                                                                             np.mean(support_loss), acc, prec, rec, f1))

                        # Outer loop
                        #query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                        query_loss, query_em = [], []
                        query_set = []

                        if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0:
    #                     if n_steps == 10:
                            for _ in range(replay_steps):
                                query_x, query_y = self.memory.read_batch(batch_size=mini_batch_size) # THIS SHOULD BE 1 else we need to do paddings?!?!? -- dicriminatory no problem since dimension1. but if qa will have problem with dims!!
    #                             print(query_x)
    #                             print(query_y)
                                query_x = torch.as_tensor(query_x, dtype=torch.long)
                                query_y = torch.as_tensor(query_y, dtype=torch.long)

                                query_set.append((query_x, query_y))
                        else:
                            try:
                                _, query_x_len, query_x, _, query_y, query_gen_x, query_gen_y = next(iter_dataloader)

                                # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
                                query_x = query_x[0]
                                query_y = query_y[0]
                                query_x_len = query_x_len[0] # an array of query x lengths, but test batch size is only1??

                                query_set.append((query_x, query_y))
                                self.memory.write_batch(query_x, query_y)
                            except StopIteration:
                                self.logger.info('Terminating training as all the data is seen')
                                break

                        for query_x, query_y in query_set:
                            query_y = query_y.to(self.device)
                            query_x = query_x.to(self.device)
                            repr_ = self.rln(query_x)[0]
                            output = fpln(repr_)

                            qa_logits = output[0]
                            qa_loss = train_loss_fct(qa_logits.transpose(1,2), query_y)
                            loss = qa_loss

                            query_loss.append(loss.item())

                            for batch_i in range(qa_logits.size()[0]):
                                _X = qa_logits.cpu()[batch_i,query_y[batch_i] != FILL_VAL, :] # This will get  EOS too size([anslen, vocab])
                                _X = torch.argmax(_X , dim=1).tolist() # Get the max preds    >> [11, 50256] (group vocab to max arg)
                                _X = list(filter(lambda x: x != -1, _X))[:-1]  # remove eos   >> [11]
                                X = self.rln.tokenizer.decode(_X) # >> [',']
                                self.logger.info(f"[ERROR_ANALYSIS] Predicted Token {_X} Answer {[X]}") # EDIT: Make this List so i can see spaces


                                _Y = query_y.cpu()[batch_i].tolist()  # >> [-1, -1, -1 ,... , 548, 3967, 50256]
                                _Y = list(filter(lambda x: x != -1, _Y))[:-1]  # remove eos & fillval >> [548, 3967]
                                print(_Y)
                                Y = ' '.join([str(y) for y in _Y]).split(str(self.rln.special_token_ids["pad_token"])) # >> ['548 3967']
                                print(Y)
                                Y = [self.rln.tokenizer.decode(list(map(int, y.split()))) for y in Y] # ['positive']
                                self.logger.info(f"[ERROR_ANALYSIS] Actual Token {_Y} Answer {Y}")

                                # SOMEHOW THIS FAILS X = qa_logits.cpu()[batch_i,query_y[batch_i] != FILL_VAL, :] 
                                # IndexError: index 1 is out of bounds for dimension 0 with size 1
                                # So I'll just get one of it, and break out
                                # I think this errors because of the last batch not having complete batch.
                                break


                            #pred = models.utils.make_prediction(output.detach())

                            #acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())
    #                         query_acc.append(acc)
    #                         query_prec.append(prec)
    #                         query_rec.append(rec)
    #                         query_f1.append(f1)
                            #  Predict vs Actual [' positive', [' very positive']]
                            # somehow X requires non list, and Y in a list?!?
                            #  EDIT: I changed *X to X since I made it not a list.
                            # EM requires X= ['negative', 'negative'] Y = [['negative], ['negative']] -> added lists
                            em = computeEM([X], [Y])
                            print(X,Y)
                            print("EM", em)
                            query_em.append(em)

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

                        print(f"number of em in the query_em {len(query_em)}")

                        self.logger.info('Episode {} query set: Loss = {:.4f}, em = {:.4f}'.format(episode_id + 1,
                                                                                np.mean(query_loss), np.mean(query_em)))


                        # Add loss to episode loss
                        episode_loss.append(np.mean(query_loss))

                        ### END Meta-Learning Phase ###
                        n_steps += 1
                        episode_id += 1

                        toc_BATCH = time.time() - tic_BATCH

                        torch.cuda.empty_cache()
    #                     if n_steps%10 == 0:
    #                         logger.info(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss_item:.5f}  Batch: {n_inputs}')
    #                         print(f'{RUN_ID} {_tasks[0]} Steps: {n_steps}/{len(train_qadata)//args.train_batch_size} Episode {n_steps}: Loss: {loss_item:.5f}  Batch: {n_inputs}')
    #                         logger.info(f'[TIME] BATCH {RUN_ID} {_tasks[0]} {toc_BATCH}')    

            toc_TASK = time.time() - tic_TASK
    
            MODEL_SAVE_LOC = os.path.join(self.model_dir, f'{dataname}.model')
            LOSS_SAVE_LOC = os.path.join(self.model_dir, f'{dataname}_loss.pickle')
            MEMORY_SAVE_LOC = os.path.join(self.model_dir, f'{dataname}_memory.pickle')
            
            self.save_model(MODEL_SAVE_LOC)
            self.logger.info(f'{dataname} Done Saving Model at {MODEL_SAVE_LOC}')
            self.logger.info(f'[TIME] TASK {dataname} {toc_TASK}')
            
            # ALSO NEED TO SAVE EPISODIC MEMORY!
            pickle.dump( self.memory, open( MEMORY_SAVE_LOC, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump( episode_loss, open( LOSS_SAVE_LOC, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )
                

    
    # For Testing Code
    # This code is almost identical to training for easiness
    # just make it only last dataset, no need to load the model_dir!!!
    # load model_dir will be in another file
    def testing(self, data_order, **kwargs):
        self.logger.info("Starting Testing code!!!")
        
        dataname_load = DATA_MAP[data_order[-1]]
        data_order_names = [DATA_MAP[i] for i in data_order]

        # For every task in the task order
        for data_eval_idx in data_order:
            tic_TASK = time.time()
            
            dataname_eval = DATA_MAP[data_eval_idx]  # This gets the name ag/yahoo/yelp......
            self.logger.info(f"Starting evaluating Last task {dataname_load} on {dataname_eval}")
            
            score_dict = {k:None for k in data_order_names}
            
            self.test_one_to_one(dataname_load, dataname_eval, score_dict, **kwargs)
            
            self.logger.info("score: {}".format(score_dict))
            with open(os.path.join(self.model_dir, f"metrics-{dataname_load}.json"),"w") as f:
                json.dump(score_dict, f)

    
    def test_one_to_one(self, dataname, dataname_eval, score_dict, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        # Some random thing from lamol dont know if can remove or not
        min_n_steps = kwargs.get('min_n_steps')
        min_batch_size = kwargs.get('min_batch_size')
        
        
        _tic_TASK = time.time() # INSIDE
        
        self.logger.info("start to test { task: %s (load) %s (eval)}" % (dataname, dataname_eval))

        self.rln.eval()
        self.pln.train()
        
        ######  Optimizer ######
        # This needs to be located here!! else the PLN weights (in higher) will make error.
        # Because pln is always changing weights through changing rln.
        # Params for Meta adaptation
        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)
        
        # Get Dataset and DataLoader
        # Load the datasets - Moving load datasets here!!!
        # Test Dataset : Support (Train QAData) Query (Test QAData)
        tic_DATALOAD = time.time()
        self.logger.info(f"Loading the dataset {dataname_eval}")
        
        
        test_dir    = datasets.utils_knight.getTaskDict(self.data_dir)[dataname_eval]["test"]
        test_qadata = QADataset(test_dir, 
                                 "test", 
                                 self.rln.special_token_ids[dataname_eval], 
                                 self.rln.special_token_ids, 
                                 self.rln.tokenizer,
                                 [], **kwargs).sort()
        if dataname_eval in ['wikisql','woz.en','multinli.in.out']:
            test_qadata.sort_by_index()
            
        max_a_len = test_qadata.max_a_len
        n_examples = len(test_qadata)
        
        test_dataloader = create_dataloader(test_qadata, "test", self.rln.special_token_ids, **kwargs)

        toc_DATALOAD = time.time() - tic_DATALOAD
        self.logger.info(f'Finished loading the dataset {dataname} in {toc_DATALOAD/60} mins')
        self.logger.info("len of test dataset: {}".format(n_examples))
        
        # test loss function - here since tokens weight may die! (change at new tokens)
        test_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=self.rln.tokens_weight)

        ##### Stream from that dataset's dataloader #####
        iter_dataloader = iter(test_dataloader)
        n_steps = 0

        episode_id = 0
        episode_loss = []
        
        
        need_process = OrderedDict()
        # qa_results is qa_results[cnt]
        qa_results = [0 for _ in range(n_examples)]
        # All pasts is shape all_pasts[layer_id][cnt]
        all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
        # max_tot_lens is qa_results[cnt]
        max_tot_lens = [0 for _ in range(n_examples)]

        cnt = 0
        
        while True:
            tic_BATCH = time.time()

            self.inner_optimizer.zero_grad()
            support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []

            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpln, diffopt):

                # Inner loop - make support set equal size to the updates
                # Since OML shows that using SGD update on single batches are better!!!
                support_set = []
                task_predictions, task_labels = [], []
                

                for _ in range(updates):
                    support_x, support_y = self.memory.read_batch(batch_size=mini_batch_size) # THIS SHOULD BE 1 else we need to do paddings?!?!? -- dicriminatory no problem since dimension1. but if qa will have problem with dims!!
                    
#                     support_x, support_y = [[[26979, 23571, 34623, 10110,    13,  4280,  1384,   284,  1949,  1223,
#            649,  1909,   986,   314,  6635, 13721,   326,  2551,    13,   775,
#           1392,   734,  1180,  3392,   290,   484,   389, 12361,   407,   284,
#           3068,   484,  1718,  1160,  2431,   284,   651,   503,   290,   612,
#            547,   691,   734,   584,  8893,   612,    13,   314,  1839,   470,
#            307, 10013,   612,   757,   986, 20498,  3792,   428,  6827,   845,
#           4633,    11,  4633,    11,  8500,    11,  3967,    11,   393,   845,
#           3967,    30, 50257,   548,  4633]], [[   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
#             -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
#             -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
#             -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
#             -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
#             -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
#             -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
#             -1,    -1,   548,  4633, 50256]]]
                    
                    support_x = torch.as_tensor(support_x, dtype=torch.long)
                    support_y = torch.as_tensor(support_y, dtype=torch.long)
                        
                n_inputs = sum(_cqa.shape[0] for _cqa in support_x)
                    
                for support_x, support_y in support_set:
                    support_y = support_y.to(self.device)
                    support_x = support_x.to(self.device)
                    repr_ = self.rln(support_x)[0]                                  # pass through rln (GPT2)
                    output = fpln(repr_)                                          # use functional (linear) pln passthrough

                    qa_logits = output[0]
                    qa_loss = train_loss_fct(qa_logits.transpose(1,2), support_y)
                    loss = qa_loss
                    diffopt.step(loss)
                    support_loss.append(loss.item())

                
                # Outer loop
                #query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                query_loss, query_em = [], []
                query_set = []
                
                try:
                    _, query_x_len, query_x, _, query_y, query_gen_x, query_gen_y = next(iter_dataloader)

                    # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
                    query_x = query_x[0]
                    query_y = query_y[0]

                    query_set.append((query_x, query_y))
                except StopIteration:
                    self.logger.info('Terminating testing as all the data is seen')
                    break

                for query_x, query_y in query_set:
                    query_y = query_y.to(self.device)
                    query_x = query_x.to(self.device)
                    with torch.no_grad():
                        repr_ = self.rln(query_x)[0]
                        output = fpln(repr_)
                        qa_logits = output[0]
                        pasts = output[1]
                        
                        next_logits = qa_logits[range(n_inputs), query_x_len-1, :] / args.temperature_qa
                        next_tokens = logits_to_tokens(next_logits).cpu()
                        
                        # Maybe this is not needed in testing since n_inputs is only 1??
                        for batch_i in range(n_inputs):
                            # max total length = max answer length + length of cq
                            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1] 
                            # add the cq of that particular batch to qa_results (Change it to cpu first!)
                            qa_results[cnt] = query_x.cpu()[batch_i][:query_x_len[batch_i]]
                            
                            # If the next tokens is not eos
                            if next_tokens[batch_i] != self.rln.special_token_ids["eos_token"]:
                                # Concat the result
                                qa_results[cnt] = torch.cat((qa_results[cnt], next_tokens[batch_i]))
                                # if the length is not max yet -> MAXTOT 225 1024
                                if len(qa_results[cnt]) not in [max_tot_lens[cnt], args.max_len]:
                                    # Append need_process of that cnt
                                    need_process.update([[cnt, None]])
                                    # Update all pasts
                                    for layer_id in range(MODEL_CONFIG.n_layer):
                                        all_pasts[layer_id][cnt] = pasts[layer_id][:, batch_i, ..., :query_x_len[batch_i], :]

                            # Try sample_sequence here! it will get all need_process (should be only 1 batch, and generate all!)
                            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens, **kwargs)
                            
                            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predicted Answer {TOKENIZER.decode(qa_results[cnt].tolist())}")
                            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predicted Tokens {qa_results[cnt].tolist()[query_x_len[batch_i]:]}")
                            # Do the score calculation here
                            # The answer of that particular batch to list
                            # EDIT
                            if task_eval in ['wikisql','woz.en']:
                                Y = test_qadata.answers[cnt]
                            else:
                                Y = query_y[batch_i].tolist()
                                Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos from the answer
                                logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Actual Tokens {Y}")
                                Y = ' '.join([str(y) for y in Y]).split(str(self.rln.special_token_ids["pad_token"]))
                                Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]

                            # Change the QA Results to a decoded version of real answer and predicted answer
                            qa_results[cnt] = [TOKENIZER.decode(qa_results[cnt].tolist()[query_x_len[batch_i]:]), Y]
                            print(f"Predict vs Actual {cnt}/{n_examples}", qa_results[cnt])
                            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Actual Answer {Y}")
                            logger.info(f"[ERROR_ANALYSIS] {task_eval} {cnt}/{n_examples} Predict vs Actual {qa_results[cnt]}")

                            cnt += 1
                        n_steps += 1
                        
                        
                        print("qa_logits SIZE", qa_logits.size())
                        print("output[1] SIZE", output[1].size())
                        
                        raise Exception("BREAKPOINT")
                        
                        qa_loss = train_loss_fct(qa_logits.transpose(1,2), query_y)
                    loss = qa_loss

                    query_loss.append(loss.item())

                    ## TEST ##
                    for batch_i in range(qa_logits.size()[0]):
                        _X = qa_logits.cpu()[batch_i,query_y[batch_i] != FILL_VAL, :] # This will get  EOS too size([anslen, vocab])
                        _X = torch.argmax(_X , dim=1).tolist() # Get the max preds    >> [11, 50256] (group vocab to max arg)
                        _X = list(filter(lambda x: x != -1, _X))[:-1]  # remove eos   >> [11]
                        X = self.rln.tokenizer.decode(_X) # >> [',']
                        self.logger.info(f"[ERROR_ANALYSIS] Predicted Token {_X} Answer {[X]}") # EDIT: Make this List so i can see spaces


                        _Y = query_y.cpu()[batch_i].tolist()  # >> [-1, -1, -1 ,... , 548, 3967, 50256]
                        _Y = list(filter(lambda x: x != -1, _Y))[:-1]  # remove eos & fillval >> [548, 3967]
                        print(_Y)
                        Y = ' '.join([str(y) for y in _Y]).split(str(self.rln.special_token_ids["pad_token"])) # >> ['548 3967']
                        print(Y)
                        Y = [self.rln.tokenizer.decode(list(map(int, y.split()))) for y in Y] # ['positive']
                        self.logger.info(f"[ERROR_ANALYSIS] Actual Token {_Y} Answer {Y}")

                        # SOMEHOW THIS FAILS X = qa_logits.cpu()[batch_i,query_y[batch_i] != FILL_VAL, :] 
                        # IndexError: index 1 is out of bounds for dimension 0 with size 1
                        # So I'll just get one of it, and break out
                        # I think this errors because of the last batch not having complete batch.
                        break
                    ## TEST ##

                    #pred = models.utils.make_prediction(output.detach())
                    #acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())

                    #  Predict vs Actual [' positive', [' very positive']]
                    # somehow X requires non list, and Y in a list?!?
                    #  EDIT: I changed *X to X since I made it not a list.
                    em = computeEM(X, Y)
                    query_em.append(em)


                print(f"number of em in the query_em {len(query_em)}")

                self.logger.info('Episode {} query set: Loss = {:.4f}, em = {:.4f}'.format(episode_id + 1,
                                                                        np.mean(query_loss), np.mean(query_em)))


                # Add loss to episode loss
                episode_loss.append(np.mean(query_loss))

                ### END Meta-Learning Phase ###
                n_steps += 1
                episode_id += 1

                toc_BATCH = time.time() - tic_BATCH

                torch.cuda.empty_cache()

    # This is different from the one above because it is run on loaded models.
    # will be called separately , and load models separately for each test
    def full_test(self, data_order, **kwargs):
        self.logger.info("Starting Full Testing code!!! 5x5")

        data_order_names = [DATA_MAP[i] for i in data_order]
        
        # For the data to be loaded (load model)
        # Get new gen tokens, and adjust tokenizer
        for data_load in data_order_names:
            model_path = os.path.join(self.model_dir, f"{data_load}.model")
            
            ##### Start training on task_id #####
            gen_token = datasets.utils_knight.get_gen_token(data_load)
            self.rln.tokenizer.add_tokens([gen_token])
            self.rln.tokenizer.save_pretrained(self.model_dir)
            self.rln.special_tokens[data_load] = gen_token
            self.rln.special_token_ids[data_load] = self.rln.tokenizer.convert_tokens_to_ids(gen_token)
            self.logger.info('gen token = {} , gen token id = {}'.format(gen_token, self.rln.special_token_ids[data_load]))
            self.rln.config.vocab_size = len(self.rln.tokenizer)
            if len(self.rln.tokenizer) != self.rln.tokens_weight.shape[0]:
                self.rln.tokens_weight = torch.cat((self.rln.tokens_weight, torch.ones([1]).cuda()))
            # Resize Token Embeddings after special tokens are appended
            self.rln.resize_token_embeddings(len(self.rln.tokenizer))
            self.pln.tie_weights(self.rln.get_input_embeddings()) # tie weights again if needed
            
            # Load the actual model from the checkpoint
            tic_LOADMODEL = time.time()
            self.load_model(model_path, map_location='cuda:0')
            self.pln.tie_weights(self.rln.get_input_embeddings()) # tie weights again if needed
            toc_LOADMODEL = time.time() - tic_LOADMODEL
            self.logger.info(f'Finished loading the model {model_path} in {toc_LOADMODEL} s')
            
            ######  Optimizer ######
            # This needs to be located here!! else the PLN weights (in higher) will make error.
            # Because pln is always changing weights through changing rln.
            # Params for Meta adaptation
            inner_params = [p for p in self.pln.parameters() if p.requires_grad]
            self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)
            
            self.logger.info(f"Task: {data_load}")
            score_dict = {k:None for k in data_order_names}
            
            # For the data to be tested against
            for data_eval in data_order_names:
                self.logger.info(f"Starting evaluating task {data_load} on {data_eval}")
                self.test_one_to_one_full(data_load, data_eval, score_dict, **kwargs)
            self.logger.info("score: {}".format(score_dict))

            with open(os.path.join(self.model_dir, f"metrics-{data_load}.json"),"w") as f:
                json.dump(score_dict, f)
                
    # Similar to test_one_to_one up top, but for full version
    def test_one_to_one_full(self, data_load, data_eval, score_dict, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        # Some random thing from lamol dont know if can remove or not
        min_n_steps = kwargs.get('min_n_steps')
        min_batch_size = kwargs.get('min_batch_size')
        temperature_qa = kwargs.get('temperature_qa')
        max_length = kwargs.get('max_length')
        
        _tic_TASK = time.time() # INSIDE
        
        self.logger.info("start to test { task: %s (load) %s (eval)}" % (data_load, data_eval))

        self.rln.eval()
        self.pln.train()

        # Get Dataset and DataLoader
        # Load the datasets - Moving load datasets here!!!
        # Test Dataset : Support (Train QAData) Query (Test QAData)
        tic_DATALOAD = time.time()
        self.logger.info(f"Loading the dataset {data_eval}")
        
        test_dir    = datasets.utils_knight.getTaskDict(self.data_dir)[data_eval]["test"]
        test_qadata = QADataset(test_dir, 
                                 "test", 
                                 self.rln.special_token_ids[data_load],  # sometimes data_eval will not be instantiated yet?!? but no use?
                                 self.rln.special_token_ids, 
                                 self.rln.tokenizer,
                                 [], **kwargs).sort()
        if data_eval in ['wikisql','woz.en','multinli.in.out']:
            test_qadata.sort_by_index()
            
        max_a_len = test_qadata.max_a_len
        n_examples = len(test_qadata)
        
        test_dataloader = create_dataloader(test_qadata, "test", self.rln.special_token_ids, **kwargs)

        toc_DATALOAD = time.time() - tic_DATALOAD
        self.logger.info(f'Finished loading the dataset {data_eval} in {toc_DATALOAD/60} mins')
        self.logger.info("len of test dataset: {}".format(n_examples))
        
        # Also load the memory !
        memory_path = os.path.join(self.model_dir, f'{data_load}_memory.pickle' )
        with open(memory_path, 'rb') as f:
            self.memory = pickle.load(f)
        
        # test loss function - here since tokens weight may die! (change at new tokens)
        test_loss_fct = CrossEntropyLoss(ignore_index=FILL_VAL, weight=self.rln.tokens_weight)

        ##### Stream from that dataset's dataloader #####
        iter_dataloader = iter(test_dataloader)
        n_steps = 0

        episode_id = 0
        episode_loss = []
        
        need_process = OrderedDict()
        # qa_results is qa_results[cnt]
        qa_results = [0 for _ in range(n_examples)]
        # All pasts is shape all_pasts[layer_id][cnt]
        all_pasts = [[0 for _ in range(n_examples)] for __ in range(self.rln.config.n_layer)]
        # max_tot_lens is qa_results[cnt]
        max_tot_lens = [0 for _ in range(n_examples)]

        cnt = 0

        while True:
            tic_BATCH = time.time()
            self.inner_optimizer.zero_grad()

            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpln, diffopt):

                # Inner loop - make support set equal size to the updates
                # Since OML shows that using SGD update on single batches are better!!!
                support_set = []
                task_predictions, task_labels = [], []
                support_loss, support_em = [], []
                
                for _ in range(updates):
                    support_x, support_y = self.memory.read_batch(batch_size=mini_batch_size) # THIS SHOULD BE 1 else we need to do paddings?!?!? -- dicriminatory no problem since dimension1. but if qa will have problem with dims!!
                    
                    support_x = torch.as_tensor(support_x, dtype=torch.long) # Size torch.Size([1, 226]) batch and len
                    support_y = torch.as_tensor(support_y, dtype=torch.long)

                n_inputs = support_x.size()[0]
                    
                for support_x, support_y in support_set:
                    support_y = support_y.to(self.device)
                    support_x = support_x.to(self.device)
                    repr_ = self.rln(support_x)[0]                                  # pass through rln (GPT2)
                    output = fpln(repr_)                                          # use functional (linear) pln passthrough

                    qa_logits = output[0]
                    qa_loss = train_loss_fct(qa_logits.transpose(1,2), support_y)
                    loss = qa_loss
                    diffopt.step(loss)
                    support_loss.append(loss.item())

                
                # Outer loop
                query_loss, query_em = [], []
                query_set = []
                
                try:
                    query_x, query_x_len, _ , _, query_y, query_gen_x, query_gen_y = next(iter_dataloader)

                    # Since we only have 1 GPU, just use the first one, it will separate batches according to the device IDS
                    query_x = query_x[0]
                    query_y = query_y[0]
                    query_x_len = query_x_len[0] # an array of query x lengths, but test batch size is only1??

                    query_set.append((query_x, query_y))
                except StopIteration:
                    self.logger.info('Terminating testing as all the data is seen')
                    break

                for query_x, query_y in query_set:
                    query_y = query_y.to(self.device)
                    query_x = query_x.to(self.device)
                    with torch.no_grad():
                        output_rln = self.rln(query_x, use_cache=True)
                        repr_ = output_rln[0]
                        pasts_rln = output_rln[1]
                        output = fpln(repr_, use_cache=True)
                        
                        #  The [0] is a  torch.Size([1, 225, 50260]), and the [1] is tuple of size 11 (RLN), in tuple of size 2, 
                        # where one of torch.Size([1, 12, 20, 64])
                        # the [1] output is different from before! before is only tuple size 12 of torch.Size([2, 1, 12, 225, 64])
                        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L416
                        # Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
                        # value states of the self-attention and the cross-attention layer
                        qa_logits = output[0]
                        pasts_pln = output[1]
                        
                        print("qa_logits size", qa_logits.size())
                        print("pasts_rln size", len(pasts_rln), len(pasts_rln[0]), pasts_rln[0][0].size())
                        # n_inputs range(0, 226), query_x_len tensor([19])
                        next_logits = qa_logits[range(n_inputs), query_x_len-1, :] / temperature_qa  # torch.Size([1, 50261])
                        next_tokens = logits_to_tokens(next_logits, **kwargs).cpu()   # torch.Size([1, 1])
                        
                        
                        
                        # Maybe this is not needed in testing since n_inputs is only 1??
                        for batch_i in range(n_inputs):
                            # max total length = max answer length + length of cq
                            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1] 
                            # add the cq of that particular batch to qa_results (Change it to cpu first!)
                            qa_results[cnt] = query_x.cpu()[batch_i][:query_x_len[batch_i]]

                            # If the next tokens is not eos
                            if next_tokens[batch_i] != self.rln.special_token_ids["eos_token"]:
                                # Concat the result
                                qa_results[cnt] = torch.cat((qa_results[cnt], next_tokens[batch_i]))
                                # if the length is not max yet -> MAXTOT 225 1024
                                if len(qa_results[cnt]) not in [max_tot_lens[cnt], max_length]:
                                    # Append need_process of that cnt
                                    need_process.update([[cnt, None]])
                                    # Update all pasts
                                    # EDIT: Modify to move [:] in pasts? not sure correct?
                                    #       Also modify batch_i to : in the second equation
                                    for layer_id in range(self.rln.config.n_layer):
                                        all_pasts[layer_id][cnt] = tuple(_k[:, ..., :query_x_len[batch_i], :] for _k in pasts_rln[layer_id])
                                        #print(" all_pasts[layer_id][cnt][0]" , all_pasts[layer_id][cnt][0].size())
                                        #print(" all_pasts[layer_id][cnt][1]", all_pasts[layer_id][cnt][1].size())
                                        
                            
                            #print("BEFORE ", qa_results[cnt])
                            # Try sample_sequence here! it will get all need_process (should be only 1 batch, and generate all!)
                            # sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)
                            
                            # ===========START SAMPLE SEQ===========
                            print("NEED PROCESS", need_process)
                            print("CNT", cnt)
#                             if len(need_process) > 0:
#                                 #print("ALLPASTS[0][cnt]", all_pasts[0][cnt])
#                                 for layer_idx in range(11):
#                                     print(f"ALLPASTS[{layer_idx}][cnt][0]", all_pasts[layer_idx][cnt][0].size())
#                                     print(f"ALLPASTS[{layer_idx}][cnt][1]", all_pasts[layer_idx][cnt][1].size())
                            self.sample_sequence(fpln, need_process, qa_results, all_pasts, max_tot_lens, **kwargs)
                            
                            # ===========END SAMPLE SEQ===========
                            
                            #print("AFTER ", qa_results[cnt])

                            self.logger.info(f"[ERROR_ANALYSIS] {data_eval} {cnt}/{n_examples} Predicted Answer {self.rln.tokenizer.decode(qa_results[cnt].tolist())}")
                            print("queryyx len", query_x_len)
                            print("queryyx lenbatch_i", query_x_len[batch_i])
                            print(qa_results[cnt].tolist()[query_x_len[batch_i]-1:])

                            self.logger.info(f"[ERROR_ANALYSIS] {data_eval} {cnt}/{n_examples} Predicted Tokens {qa_results[cnt].tolist()[query_x_len[batch_i]:]}")

                            # Do the score calculation here
                            # The answer of that particular batch to list
                            if data_eval in ['wikisql','woz.en']:
                                Y = test_qadata.answers[cnt]
                            else:
                                Y = query_y[batch_i].tolist()
                                Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos from the answer
                                self.logger.info(f"[ERROR_ANALYSIS] {data_eval} {cnt}/{n_examples} Actual Tokens {Y}")
                                Y = ' '.join([str(y) for y in Y]).split(str(self.rln.special_token_ids["pad_token"]))
                                Y = [self.rln.tokenizer.decode(list(map(int, y.split()))) for y in Y]

                            # Change the QA Results to a decoded version of real answer and predicted answer
                            qa_results[cnt] = [self.rln.tokenizer.decode(qa_results[cnt].tolist()[query_x_len[batch_i]:]), Y]
                            print(f"Predict vs Actual {cnt}/{n_examples}", qa_results[cnt])
                            self.logger.info(f"[ERROR_ANALYSIS] {data_eval} {cnt}/{n_examples} Actual Answer {Y}")
                            self.logger.info(f"[ERROR_ANALYSIS] {data_eval} {cnt}/{n_examples} Predict vs Actual {qa_results[cnt]}")
                            
                            score = compute_metrics([qa_results[cnt]])
                            self.logger.info(f"[ERROR_ANALYSIS] em {score['em']}")

                            cnt += 1
                        n_steps += 1

        _toc_TASK = time.time() - _tic_TASK
        self.logger.info(f'[TIME] TASK {(data_load, data_eval)} {_toc_TASK}')

        get_test_score(data_eval, qa_results, score_dict)
        print(score_dict)

        results_path = os.path.join(self.model_dir,f"qa_{data_load}_{data_eval}.csv")
        with open(results_path, "w",encoding="utf-8") as f:
            qa_writer = csv.writer(f,delimiter=',')
            qa_writer.writerow(["y","pred"])
            for pred, y in qa_results:
                # EDIT 
                if data_eval == 'wikisql': 
                    y = y["answer"]
                elif data_eval == 'woz.en': 
                    y = y[1]
                qa_writer.writerow([y,pred]) 
                
                
    def sample_sequence(self, fpln, need_process, qa_results, all_pasts, max_tot_lens, **kwargs):
        test_batch_size = kwargs.get('test_batch_size')
        temperature_qa = kwargs.get('temperature_qa')
        max_length = kwargs.get('max_length')
        

        while len(need_process) > 0:
            first_id = next(iter(need_process))
            shortest_len = len(qa_results[first_id])
            decode_batch_size = test_batch_size
            it = iter(need_process)
            stop = False
            remove_ids = []
            while not stop:
                batch_ids, input_ids, past = [], [], [[] for _ in range(self.rln.config.n_layer)]
                while True:
                    try:
                        cur_id = next(it)
                        if len(qa_results[cur_id]) > shortest_len:
                            stop = True
                            break
                        batch_ids.append(cur_id)
                        input_ids.append(qa_results[cur_id][-1:])
                        for layer_id in range(self.rln.config.n_layer): # Change to extend since it's now tuple
                            past[layer_id].extend(all_pasts[layer_id][cur_id])
                        if len(input_ids) == decode_batch_size:
                            break
                    except StopIteration:
                        stop = True
                        break

                
                n_inputs = len(input_ids)
                if n_inputs == 0:
                    break
                input_ids = torch.stack(input_ids)
#                                     for layer_id in range(self.rln.config.n_layer):
#                                         past[layer_id] = torch.stack(past[layer_id], dim=1)
#                 print("PAST OF Layer [0]")
#                 print(len(past[0]))
#                 print(past[0][0].size())
#                 print(past[0][1].size())
                _output_rln = self.rln(input_ids=input_ids.cuda(), past_key_values=past, use_cache=True)
                _repr_ = _output_rln[0]
                _pasts_rln = _output_rln[1]

                _output = fpln(_repr_)
                _qa_logits = _output[0]

                _next_logits = _qa_logits[..., -1, :] / temperature_qa
                _next_tokens = logits_to_tokens(_next_logits).cpu()

                for i, cur_id in enumerate(batch_ids):
                    if _next_tokens[i] == self.rln.special_token_ids["eos_token"]:
                        remove_ids.append(cur_id)
                    else:
                        qa_results[cur_id] = torch.cat((qa_results[cur_id], _next_tokens[i]))
                        if len(qa_results[cur_id]) in [max_tot_lens[cur_id], max_length]:
                            remove_ids.append(cur_id)
                        else:
                            # Change this to not select batch anymore.  - it needs batch dimension or will error!
                            for layer_id in range(self.rln.config.n_layer):  
                                all_pasts[layer_id][cur_id] = _pasts_rln[layer_id]
                                #print("THIS IS ALL PASTS22", all_pasts[layer_id][cnt])
            for idx in remove_ids:
                remove_id(idx, need_process, all_pasts, self.rln.config.n_layer)

                        
                        
                        
#      def evaluate(self, dataloader, updates, mini_batch_size):

#         self.rln.eval()
#         self.pln.train()

#         support_set = []
#         for _ in range(updates):
#             text, labels = self.memory.read_batch(batch_size=mini_batch_size)
#             support_set.append((text, labels))

#         with higher.innerloop_ctx(self.pln, self.inner_optimizer,
#                                   copy_initial_weights=False,
#                                   track_higher_grads=False) as (fpln, diffopt):

#             # Inner loop
#             task_predictions, task_labels = [], []
#             support_loss = []
#             for text, labels in support_set:
#                 labels = torch.tensor(labels).to(self.device)
#                 input_dict = self.rln.encode_text(text)
#                 repr = self.rln(input_dict)
#                 output = fpln(repr)
#                 loss = self.loss_fn(output, labels)
#                 diffopt.step(loss)
#                 pred = models.utils.make_prediction(output.detach())
#                 support_loss.append(loss.item())
#                 task_predictions.extend(pred.tolist())
#                 task_labels.extend(labels.tolist())

#             acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

#             self.logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
#                         'recall = {:.4f}, F1 score = {:.4f}'.format(np.mean(support_loss), acc, prec, rec, f1))

#             all_losses, all_predictions, all_labels = [], [], []

#             for text, labels in dataloader:
#                 labels = torch.tensor(labels).to(self.device)
#                 input_dict = self.rln.encode_text(text)
#                 with torch.no_grad():
#                     repr = self.rln(input_dict)
#                     output = fpln(repr)
#                     loss = self.loss_fn(output, labels)
#                 loss = loss.item()
#                 pred = models.utils.make_prediction(output.detach())
#                 all_losses.append(loss)
#                 all_predictions.extend(pred.tolist())
#                 all_labels.extend(labels.tolist())

#         acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
#         self.logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
#                     'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))

#         return acc, prec, rec, f1


    