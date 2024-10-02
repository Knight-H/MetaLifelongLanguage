from datasets.text_classification_dataset import MAX_TRAIN_SIZE, MAX_VAL_SIZE
import logging
import os, time
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np

import torch

import datasets.utils
from models.cls_agem import AGEM
from models.cls_anml import ANML
from models.cls_baseline import Baseline
from models.cls_maml import MAML
from models.cls_oml import OML
from models.cls_replay import Replay
from models.cls_oml2 import OML2

# python test_text_cls2.py --order 1 --learner oml2 --model gpt2 --max_length 1024 --run_id 20220529T095601_1_oml2gpt2 --test

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
    parser.add_argument('--lr', type=float, help='Learning rate (only for the baselines)', default=3e-5)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='bert')
    parser.add_argument('--learner', type=str, help='Learner method', default='oml')
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=1)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=1.0)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for the input', default=448)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory', default=0.01)
    parser.add_argument('--replay_every', type=int, help='Number of data points between replay', default=9600)
    # Something from Lamol
    parser.add_argument("--token_weight", type=float, default=5)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    parser.add_argument("--tokenize_n_cpu", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=1) # MAKE THIS 1 FOR OML!!!! since we do it one at a time
    parser.add_argument("--train_batch_size", type=int, default=1) # MAKE THIS 1 FOR OML!!!! since we do it one at a time
    parser.add_argument("--n_gpus", type=int, default=1)
    # Generation Sample Sequence
    parser.add_argument("--temperature_lm", type=float, default=1.0)
    parser.add_argument("--temperature_qa", type=float, default=1.0)
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_k_qa", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.)
    parser.add_argument("--top_p_qa", type=float, default=0.)
    # Directories
    parser.add_argument("--data_dir", type=str, default="/data/lamol_data")
    parser.add_argument("--run_id", type=str, default="20220529T095601_1_oml2gpt2")
    parser.add_argument("--model_base_dir", type=str, default="/data/model_runs")
    # Test or Train
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--10k", action='store_true')
    args = parser.parse_args()

    # Set base path
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Run ID
    RUN_ID = args.run_id
    
    # Make Model Dir and Create if not exist
    model_dir = os.path.join(args.model_base_dir, RUN_ID)
    args.run_id = RUN_ID
    args.model_dir = model_dir
    # Set logger to both the console and the file
    # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    # https://stackoverflow.com/questions/6614078/logging-setlevel-how-it-works
    logger = logging.getLogger('ContinualLearningLog')
    logger.setLevel(level=logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fileHandler = logging.FileHandler(f'{model_dir}/{RUN_ID}.{"test" if args.test else "log"}')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    logger.propagate = False
    
    logger.info(f"Starting Test for RUN_ID {RUN_ID}")
    logger.info(f"[TIME] Start Test {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}")
    tic_RUN = time.time()
    logger.info('Using configuration: {}'.format(vars(args)))

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.learner == 'sequential':
        learner = Baseline(device=device, n_classes=n_classes, training_mode='sequential', **vars(args))
    elif args.learner == 'multi_task':
        learner = Baseline(device=device, n_classes=n_classes, training_mode='multi_task', **vars(args))
    elif args.learner == 'agem':
        learner = AGEM(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'replay':
        learner = Replay(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'maml':
        learner = MAML(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'oml':
        learner = OML(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'oml2':
        learner = OML2(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'anml':
        learner = ANML(device=device, n_classes=n_classes, **vars(args))
    else:
        raise NotImplementedError
    logger.info('Using {} as learner'.format(learner.__class__.__name__))

    # Testing
    logger.info('----------Testing on test set starts here----------')
    learner.full_test(dataset_order_mapping[args.order], **vars(args))
    
    toc_RUN = time.time() - tic_RUN
    logger.info(f"[TIME] End Test {RUN_ID} at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
