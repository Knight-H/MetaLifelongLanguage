from datasets.text_classification_dataset import MAX_TRAIN_SIZE, MAX_VAL_SIZE
import logging
import os
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

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')

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
    parser.add_argument('--lr', type=float, help='Learning rate (only for the baselines)', default=3e-5)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='bert')
    parser.add_argument('--learner', type=str, help='Learner method', default='oml')
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=16)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=1.0)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for the input', default=448)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory', default=0.01)
    parser.add_argument('--replay_every', type=int, help='Number of data points between replay', default=9600)
    args = parser.parse_args()
    logger.info('Using configuration: {}'.format(vars(args)))

    # Set base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the datasets
    logger.info('Loading the datasets')
    train_datasets, val_datasets, test_datasets = [], [], []
    for dataset_id in dataset_order_mapping[args.order]:
        train_dataset, test_dataset = datasets.utils.get_dataset(base_path, dataset_id)
        logger.info('Loaded {}'.format(train_dataset.__class__.__name__))
        train_dataset = datasets.utils.offset_labels(train_dataset)
        test_dataset = datasets.utils.offset_labels(test_dataset)
        train_dataset, val_dataset = datasets.utils.get_train_val_split(dataset=train_dataset,
                                                                        train_size=MAX_TRAIN_SIZE,
                                                                        val_size=MAX_VAL_SIZE)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
    logger.info('Finished loading all the datasets')

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

    # Training
    model_file_name = learner.__class__.__name__ + '-' + str(datetime.now()).replace(':', '-').replace(' ', '_') + '.pt'
    model_dir = os.path.join(base_path, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    logger.info('----------Training starts here----------')
    learner.training(train_datasets, **vars(args))
    learner.save_model(os.path.join(model_dir, model_file_name))
    logger.info('Saved the model with name {}'.format(model_file_name))

    # Testing
#     logger.info('----------Testing on val set starts here----------')
#     learner.testing(test_datasets, **vars(args))

    # Testing
    logger.info('----------Testing on test set starts here----------')
    learner.testing(test_datasets, **vars(args))
