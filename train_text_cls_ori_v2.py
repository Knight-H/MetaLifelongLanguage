from datasets.text_classification_dataset import MAX_TRAIN_SIZE, MAX_VAL_SIZE
import logging
import os, time
import random
import pickle
from argparse import ArgumentParser
from datetime import datetime

import numpy as np

import torch

import datasets.utils
# from models.cls_agem import AGEM
# from models.cls_anml import ANML
from models.cls_baseline import Baseline
# from models.cls_maml import MAML
# from models.cls_oml_ori_v2 import OML
from models.cls_replay import Replay
# from models.cls_oml2 import OML2

# logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger('ContinualLearningLog')

# python train_text_cls.py --order 1 --learner oml2 --model gpt2


if __name__ == '__main__':

    # Define the ordering of the datasets
    dataset_order_mapping = {
        1: [2, 0, 3, 1, 4],
        2: [3, 4, 0, 1, 2],
        3: [2, 4, 1, 3, 0],
        4: [0, 2, 1, 4, 3]
    }
    n_classes = 33

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--order', type=int, help='Order of datasets', required=True)
    parser.add_argument('--n_epochs', type=int, help='Number of epochs (only for MTL)', default=1)
    parser.add_argument('--partials', type=int, help='Number of partials in the order (only for MTL)')
    parser.add_argument('--early_stopping', action='store_true', help='Whether to not specify epoch and do early stopping (only for MTL)')
    parser.add_argument('--lr', type=float, help='Learning rate (only for the baselines)', default=3e-5)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='bert')
    parser.add_argument('--learner', type=str, help='Learner method', default='oml')
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=16)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    # This doesn't matter for MeLSTA because we already fix the buffer at the back. this is just frequency of deciding to write or not. 
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=1.0)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for the input', default=448)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory', default=0.01)
    parser.add_argument('--replay_every', type=int, help='Number of data points between replay', default=9600)
    parser.add_argument('--model_dir', type=str, help='Directory to save the model', default='/data/model_runs/original_oml')
    parser.add_argument('--cache_dir', type=str, help='Directory to save cache data', default='/data/omler_data/tmp')
    parser.add_argument('--log_freq', type=int, help='Log Frequency', default=500)
    # Selective Replay
    parser.add_argument('--selective_replay', action='store_true', help='Selective Replay or normal Replay')
    parser.add_argument('--all_query_replay', action='store_true', help='Query set will use replay for sampling')
    # ER Score-Based Replay
    parser.add_argument('--score_based_replay', action='store_true', help='score based replay in ER')
    parser.add_argument('--validation_split', type=float, help='Split percentage of ER', default=0.)
    # Task Aware
    parser.add_argument('--task_aware', action='store_true', help='ER will be task aware')
    # Reverse Support
    parser.add_argument('--reverse_support', action='store_true', help='Similar to all query, but make everything support instead')
    # Curriculum Replay
    parser.add_argument('--curriculum_replay', action='store_true', help='Instead of replaying on random task, just replay on curriculum of the current task')
    # Custom PLN
    parser.add_argument('--pln', type=str, help='PLN model to use', default='1fc')
    # Adding Adapter 
    parser.add_argument('--adapter', action='store_true', help='PLN will also add adapter')
    parser.add_argument('--adapter_type', type=str, help='Adapter model to use', default='lora')
    parser.add_argument('--inner_adapter_lr', type=float, help='Inner-loop learning rate', default=1e-4) # This doesn't work yet since we merge adapters....
    # Early Stopping
    # parser.add_argument('--early_stopping', action='store_true', help='ER will do train/validation split and will check a-n every adaptation')
    args = parser.parse_args()
    
    if not args.adapter:
        from models.cls_oml_ori_v2 import OML
    else:
        from models.cls_oml_adapter import OML
    
    # Log to save
    log_dir = args.model_dir
    log_file_name = f"{args.learner.upper()[:3]}-order{args.order}-{str(datetime.now()).replace(':', '-').replace(' ', '_')}.log"
    LOG_LOC = os.path.join(log_dir, log_file_name)
    # Set logger to both the console and the file
    # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    # https://stackoverflow.com/questions/6614078/logging-setlevel-how-it-works
    logger = logging.getLogger('ContinualLearningLog')
    logger.setLevel(level=logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fileHandler = logging.FileHandler(LOG_LOC)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.propagate = False
    
    logger.info(f"[TIME] Start Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    logger.info('Using configuration: {}'.format(vars(args)))
    
    ## Added db caching for ease of load
    use_db_cache = True

    # Set base path
    #base_path = os.path.dirname(os.path.abspath(__file__))

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the datasets
    logger.info('Loading the datasets')
    train_datasets, val_datasets, test_datasets = [], [], []
    data_order = dataset_order_mapping[args.order]
    if args.partials:
        data_order = data_order[:args.partials]
    for dataset_id in data_order:
        
        # KNIGHT EDIT TO MAKE IT LOAD TRAINING DATA FAST!
        train_dataset_file = os.path.join(args.cache_dir, f"train-{dataset_id}.cache")
        if os.path.exists(train_dataset_file):
            with open(train_dataset_file, 'rb') as f:
                train_dataset = pickle.load(f)
        else:
            train_dataset = datasets.utils.get_dataset_train("", dataset_id)
            print('Loaded {}'.format(train_dataset.__class__.__name__))
            train_dataset = datasets.utils.offset_labels(train_dataset)
            pickle.dump(train_dataset, open( train_dataset_file, "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
            print("Type: ", type(train_dataset))
            print(f"Pickle saved at {train_dataset_file}")
        
        
        #train_dataset, test_dataset = datasets.utils.get_dataset(base_path, dataset_id)
        #logger.info('Loaded {}'.format(train_dataset.__class__.__name__))
        #train_dataset = datasets.utils.offset_labels(train_dataset)
        #test_dataset = datasets.utils.offset_labels(test_dataset)
        
        # Somehow train_dataset is 120k, so we're breaking it up into 115k (Train) and 5k (Valid)
        # And lifelong learning doesn't care about validation, so we just use validation for MTL
        train_dataset, val_dataset = datasets.utils.get_train_val_split(dataset=train_dataset,
                                                                        train_size=MAX_TRAIN_SIZE,
                                                                        val_size=MAX_VAL_SIZE)
        print("Type2: ", type(train_dataset)) # This is Torch.data.Subset
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        #test_datasets.append(test_dataset)
    logger.info('Finished loading all the datasets')

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.learner == 'sequential':
        learner = Baseline(device=device, n_classes=n_classes, training_mode='sequential', **vars(args))
    elif args.learner == 'multi_task':
        learner = Baseline(device=device, n_classes=n_classes, training_mode='multi_task', **vars(args))
#     elif args.learner == 'agem':
#         learner = AGEM(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'replay':
        learner = Replay(device=device, n_classes=n_classes, **vars(args))
#     elif args.learner == 'maml':
#         learner = MAML(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'oml':
        learner = OML(device=device, n_classes=n_classes, **vars(args))
#     elif args.learner == 'oml2':
#         learner = OML2(device=device, n_classes=n_classes, **vars(args))
#     elif args.learner == 'anml':
#         learner = ANML(device=device, n_classes=n_classes, **vars(args))
    else:
        raise NotImplementedError
    logger.info('Using {} as learner'.format(learner.__class__.__name__))

    # Training
    model_file_name = learner.__class__.__name__ + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pt'
    #model_dir = os.path.join(base_path, 'saved_models')
    #os.makedirs(model_dir, exist_ok=True)
    logger.info('----------Training starts here----------')
    learner.training(train_datasets, val_datasets, **vars(args)) # val_datasets are only used for MTL early stopping
    #learner.save_model(os.path.join(model_dir, model_file_name))
    #logger.info('Saved the model with name {}'.format(model_file_name))
    
    
    toc_RUN = time.time() - tic_RUN
    logger.info(f"[TIME] End Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
    

    # Testing
#     logger.info('----------Testing on val set starts here----------')
#     learner.testing(test_datasets, **vars(args))

    # Testing
    #logger.info('----------Testing on test set starts here----------')
    #learner.testing(test_datasets, **vars(args))
