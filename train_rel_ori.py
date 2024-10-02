import gc
import logging
import os, time
import random
from argparse import ArgumentParser
from datetime import datetime

import torchtext
import torch
import numpy as np

import datasets.utils
from datasets.lifelong_fewrel_dataset import LifelongFewRelDataset
# from models.rel_agem import AGEM
# from models.rel_anml import ANML
from models.rel_baseline import Baseline
# from models.rel_maml import MAML
from models.rel_oml import OML
from models.rel_melsta import MeLSTA # New!!
from models.rel_replay import Replay

# logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger('ContinualLearningLog')


if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, help='Number of epochs (only for MTL)', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate (only for the baselines)', default=3e-5)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='bert')
    parser.add_argument('--learner', type=str, help='Learner method', default='sequential')
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=4)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=1.0)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for the input', default=64)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory', default=0.01)
    parser.add_argument('--order', type=int, help='Number of task orders to run for', default=5)
    parser.add_argument('--num_clusters', type=int, help='Number of clusters to take', default=10)
    parser.add_argument('--replay_every', type=int, help='Number of data points between replay', default=1600)
    parser.add_argument('--model_dir', type=str, help='Directory to save the model', default='/data/model_runs/original_oml')
    # ~~~~~~~~~~~~ Added Custom ~~~~~~~~~~~~~~~
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
    args = parser.parse_args()
    
    # Log to save
    log_dir = args.model_dir
    log_file_name = f"{args.learner.upper()[:3]}-order{args.order}-{str(datetime.now()).replace(':', '-').replace(' ', '_')}.log"
    LOG_LOC = os.path.join(log_dir, log_file_name)
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

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Random instance independent for shuffle Index!!
    shuffle_index_generator = random.Random(args.seed)

    # Set base path
#     base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load training and validation data
    logger.info('Loading the dataset')
    data_dir = '/data/omler_data/LifelongFewRel'
    relation_file = os.path.join(data_dir, 'relation_name.txt')
    training_file = os.path.join(data_dir, 'training_data.txt')
    validation_file = os.path.join(data_dir, 'val_data.txt')
    # ie. ['fill', ['place', 'served', 'by', 'transport', 'hub'], ['mountain', 'range'], ['religion'], ['participating', 'team'], ...]
    # Note that 'fill' is the 0 index, can be ignored
    relation_names = datasets.utils.read_relations(relation_file) # List of relation names (converted to 1-based index later)
    train_data = datasets.utils.read_rel_data(training_file)
    val_data = datasets.utils.read_rel_data(validation_file)
    logger.info('Finished loading the dataset')
    # label2idx is reverse of relation_names,  where we map label (space joined relation) --> label id. ie. "work location" --> 2
    # Used for Label Aware ER -- maybe not need if we just use list as key???
    label2idx = {" ".join(relation_name): i for i,relation_name in enumerate(relation_names) if i != 0}
    args.idx2label = relation_names
    args.label2idx = label2idx

    # Load GloVe vectors
    logger.info('Loading GloVe vectors')
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    logger.info('Finished loading GloVe vectors')

    # Get relation embeddings for clustering
    relation_embeddings = datasets.utils.get_relation_embedding(relation_names, glove)
    print(relation_embeddings.shape)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate clusters
    # This essentially goes through all train_data and get label set, which is a list of 1-80 ie. [80, 25, 75, 15, 62, 74, 5, 10...] 
    relation_index = datasets.utils.get_relation_index(train_data)  
    # This uses KMeans to divide the label up into 10 disjoint clusters ie. {80: 1, 25: 5, 75: 3, 15: 1, 62: 1, 74: 1, 5: 1, 10: 2...}
    # > relation_embeddings just return a dictionary of relation_index --> Glove embedding ie. { 80: embedding, 25: embedding, ...}
    cluster_labels, relation_embeddings = datasets.utils.create_relation_clusters(args.num_clusters, relation_embeddings, relation_index)
    
    # Task Dict is essentially the reverse dictionary of cluster_labels, where cluster_idx --> relation_idx[] (Used for Selective Replay)
    # task_dict {1: [80, 15, 62, 74, 5, 64, 8, 46, 65, 49, 44, 38, 59, 33, 23, 58, 39, 14, 32, 2, 9, 48, 1], 5: [25, 11, 20, 6, 78, 31, 51, 66, 76, 30, 47, 72, 13, 77, 7, 63, 68, 27, 12, 29, 22, 60, 37, 61, 3, 36, 79, 17], 3: [75, 34], 2: [10, 35, 43, 73], 7: [69, 4, 41, 54, 50, 53, 67, 26, 24, 71, 57], 6: [28, 55, 40, 18, 21, 42], 9: [19, 52, 56], 8: [45], 0: [16], 4: [70]}
    task_dict = {}
    for relation_id, cluster_id in cluster_labels.items():
        # If want to keep as ID
        task_dict[cluster_id] = task_dict.get(cluster_id, []) + [relation_id]
        # If not, it will keep as task_key like this "7|['genre']", "7|['part', 'of']"
        # Note: This doesn't work because in LA-ER read, there is  class_id = key if not self.task_aware else int(key.split("|")[-1])
        #       You can't exactly int() a list or list a string.
        #task_dict[cluster_id] = task_dict.get(cluster_id, []) + [relation_names[relation_id]]
    args.task_dict = task_dict
    args.cluster_labels = cluster_labels
    print(task_dict)

    # Validation dataset
    if args.learner == "melsta":
        # Validation Datast v2 for Task-Aware , Separate it into the 3 clusters
        val_dataset = [LifelongFewRelDataset(vd, relation_names) for vd in datasets.utils.split_rel_data_by_clusters(val_data, cluster_labels, args.num_clusters, list(range(args.num_clusters)))]
        print(f"Val Dataset2 Length: {[len(x) for x in val_dataset]}")
        print(f"Val Dataset2 Sum: {sum([len(x) for x in val_dataset])}")
    else:
        val_dataset = LifelongFewRelDataset(val_data, relation_names)
        print(f"Val Dataset Length: {len(val_dataset)}")
    

    # Run for different orders of the clusters
    accuracies = []
    for i in range(args.order):

        logger.info('Running order {}'.format(i + 1))

        # Initialize the model
        if args.learner == 'sequential':
            learner = Baseline(device=device, training_mode='sequential', **vars(args))
        elif args.learner == 'multi_task':
            learner = Baseline(device=device, training_mode='multi_task', **vars(args))
#         elif args.learner == 'agem':
#             learner = AGEM(device=device, **vars(args))
        elif args.learner == 'replay':
            learner = Replay(device=device, **vars(args))
#         elif args.learner == 'maml':
#             learner = MAML(device=device, **vars(args))
        elif args.learner == 'oml':
            learner = OML(device=device, **vars(args))
        elif args.learner == 'melsta':
            learner = MeLSTA(device=device, **vars(args))
#         elif args.learner == 'anml':
#             learner = ANML(device=device, **vars(args))
        else:
            raise NotImplementedError
        logger.info('Using {} as learner'.format(learner.__class__.__name__))

        # Generate continual learning training data
        logger.info('Generating continual learning data')
        train_datasets, shuffle_index = datasets.utils.prepare_rel_datasets(train_data, relation_names, cluster_labels, args.num_clusters, shuffle_index_generator)
        args.shuffle_index = shuffle_index
        logger.info(f"Shuffle Index: {shuffle_index}")
        logger.info(f"Train Dataset Length: {[len(x) for x in train_datasets]}")
        logger.info('Finished generating continual learning data')

        # Training
        logger.info('----------Training starts here----------')
        model_file_name = f"{learner.__class__.__name__}-order{args.order}-id{i}-{str(datetime.now()).replace(':', '-').replace(' ', '_')}.pt"
        MODEL_LOC = os.path.join(args.model_dir, model_file_name)
        #os.makedirs(model_dir, exist_ok=True)
        learner.training(train_datasets, **vars(args))
        learner.save_model(MODEL_LOC)
        logger.info('Saved the model with name {}'.format(model_file_name))

        # Testing
        logger.info('----------Testing starts here----------')
        acc = learner.testing(val_dataset, **vars(args))
        accuracies.append(acc)

        # Delete the model to free memory
        del learner
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logger.info('Accuracy across runs = {}'.format(accuracies))
    logger.info('Average accuracy across runs: {}'.format(np.mean(accuracies)))


    toc_RUN = time.time() - tic_RUN
    logger.info(f"[TIME] End Run at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')} within {toc_RUN/3600} hours")
