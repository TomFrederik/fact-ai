from typing import Dict, Type, Optional, Any, List, Tuple

import main

import torch

import os
import json
import argparse
import itertools as it
from time import time

# dict to access optimizers by name, if we need to use different opts.
OPT_BY_NAME: Dict[str, Type[torch.optim.Optimizer]] = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam
}

MODELS = ['baseline', 'ARL', 'DRO', 'IPW(S)', 'IPW(S+Y)']
DATASETS = ['Adult', 'COMPAS', 'LSAC']
PARAMS = ['lr', 'batch_size']


def get_tabular_opt_hparams(args):
    """
    Executes grid searches of all models on all tabular datasets.
    Saves the optimal
    Args:
        args: Object from the argument parser that defines various settings of
            the model, dataset and training.
    """

    all_best_params = {dataset: {} for dataset in DATASETS}

    args.version = str(int(time()))

    for model, dataset in it.product(MODELS, DATASETS):
        
        print(f'Now running grid search for {model} on {dataset}')
        
        # set args for this grid search
        if model == 'IPW(S)':
            args.model = 'IPW'
            args.sensitive_label = False
        elif model == 'IPW(S+Y)':
            args.model = 'IPW'
            args.sensitive_label = True
        else:
            args.model = MODELS
            args.sensitive_label = False
        
        args.dataset = dataset

        # run grid_search
        auc_scores, best_params = main.main(args)

        print(f'Best params for {model} on {dataset} are {best_params}')

        # add params to dict
        all_best_params[dataset][model] = {key: best_params[key] for key in PARAMS}
        if model == 'DRO':
            all_best_params[dataset][model]['eta'] = best_params['eta']
        
        # write to disk, to ensure it is saved even if run is aborted later
        path = './optimal_hparams_TEST.json'
        with open(path) as f:
            json.dump(f)
        
    print(f'Search complete! Best params are {all_best_params}.')




    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model settings
    #parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO', 'IPW'], required=True)
    parser.add_argument('--prim_hidden', nargs='*', type=int, default=[64, 32], help='Number of hidden units in primary network')
    parser.add_argument('--adv_hidden', nargs='*', type=int, default=[], help='Number of hidden units in adversarial network')
    parser.add_argument('--k', default=2.0, type=float, help='Exponent to upweight high losses')
    parser.add_argument('--pretrained', action='store_true', help='Whether to load a pretrained dataset from torchvision where applicable')
    parser.add_argument('--adv_input', nargs='+', default=['X', 'Y'], help='Inputs to use for the adversary. Any combination of X (features), Y (labels) and S (protected group memberships)')
    parser.add_argument('--seed', default=0, type=int)

    # train settings
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--pretrain_steps', default=250, type=int)
    parser.add_argument('--opt', choices=OPT_BY_NAME.keys(), default="Adagrad", help='Name of optimizer')
    parser.add_argument('--log_dir', default='training_logs', type=str)
    parser.add_argument('--p_bar', action='store_true', help='Whether to use progressbar')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of crossvalidation folds')
    parser.add_argument('--no_grid_search', action='store_false', default=True, dest="grid_search", help='Don\'t optimize batch size and lr via gridsearch')
    #parser.add_argument('--nbr_seeds', default=2, type=int, help='Number of independent training runs') # TODO: not implemented yet
    parser.add_argument('--eval_batch_size', default=512, type=int, help='Batch size for evaluation. No effect on training or results, set as large as memory allows to maximize performance')
    parser.add_argument('--tf_mode', action='store_true', default=False, help='Use tensorflow rather than PyTorch defaults where possible. Only supports AdaGrad optimizer.')
    
    # Dataset settings
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers that are used in dataloader')
    parser.add_argument('--disable_warnings', action='store_true', help='Whether to disable warnings about mean and std in the dataset')

    # ray settings
    parser.add_argument('--num_cpus', default=1, type=int, help='Number of CPUs used for each trial')
    parser.add_argument('--num_gpus', default=0.25, type=float, help='Number of GPUs used for each trial')

    args = parser.parse_args()

    get_tabular_opt_hparams(args)