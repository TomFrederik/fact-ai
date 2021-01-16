import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import CustomDataset, CustomSubset
from arl import ARL
from dro import DRO
from ipw import IPW
from baseline_model import BaselineModel
from metrics import Logger, get_all_auc_scores

import argparse
import os

import numpy as np
from time import time
import itertools
import json
import warnings

from ray import tune

from sklearn.model_selection import KFold

print(f'Cuda available? {torch.cuda.is_available()}')


# dict to access optimizers by name, if we need to use different opts.
OPT_BY_NAME = {'Adagrad': torch.optim.Adagrad}

# obscure bug fix to get ray + slurm + ptl to cooperate
os.environ["SLURM_JOB_NAME"] = "bash"


def main(args):
    
    # set run version
    args.version = str(int(time()))
    
    # seed
    pl.seed_everything(args.seed)

    # create datasets
    dataset = CustomDataset(args.dataset, sensitive_label=args.sensitive_label, disable_warnings=args.disable_warnings)
    test_dataset = CustomDataset(args.dataset, sensitive_label=args.sensitive_label, test=True, disable_warnings=args.disable_warnings)
    
    # init config dictionary
    config = {}
    
    if args.grid_search:
        # specify search space 
        # TODO: pull this outside this function for more flexible search space?
        lr_list = [0.001, 0.01, 0.1, 1, 2, 5]
        batch_size_list = [32, 64, 128, 256, 512]
        eta_list = [0] # dummy entry for non-DRO experiments
        
        if args.model == 'DRO':
            eta_list = [0.5, 0.6, 0.7, 0.8, 0.9]    
    
        # configurations for hparam tuning
        config = {
            'lr': tune.grid_search(lr_list),
            'batch_size': tune.grid_search(batch_size_list),
            'eta': tune.grid_search(eta_list)
            }
      
        # perform n-fold crossvalidation
        kf = KFold(n_splits=args.num_folds)          
        fold_indices = list(kf.split(dataset))
        
        # set path for logging
        path = f'./grid_search/{args.model}_{args.dataset}_version_{args.version}'
        
        analysis = tune.run(
            tune.with_parameters(
                run_folds,
                args=args,
                dataset=dataset,
                fold_indices=fold_indices,
                version=args.version),
            resources_per_trial={
                'cpu': args.num_cpus,
                'gpu': args.num_gpus if torch.cuda.is_available() else 0,
            },
            config=config,
            metric='auc',
            mode='max',
            local_dir=os.getcwd(),
            name=path 
            ) 
        
        print('Best hyperparameters found were: ', analysis.best_config)
        
        # save grid search results 
        df = analysis.results_df
        df.to_csv(os.path.join(path, 'results.csv'), index=False)
        
        # set hparams for final run
        config['lr'] = analysis.best_config['lr']
        config['batch_size'] = analysis.best_config['batch_size']
        config['eta'] = analysis.best_config['eta']
        
    else:
        # set hparams for single run
        config['lr'] = args.prim_lr
        config['batch_size'] = args.batch_size
        config['eta'] = args.eta
        
        if args.seed_run:
            path = f'./{args.log_dir}/{args.dataset}/{args.model}/version_{args.seed_run_version}/seed_{args.seed}'
        else:
            path = f'./{args.log_dir}/{args.dataset}/{args.model}/version_{args.version}'
        
    # single training run
    model = train(config, args, train_dataset=dataset, test_dataset=test_dataset)
    
    # compute final test scores
    auc_scores = get_all_auc_scores(model, test_dataset, args.eval_batch_size)
        
    # print results
    print(f'Results = {auc_scores}')
    
    # save results
    with open(os.path.join(path, 'auc_scores.json'),'w') as f:
        json.dump(auc_scores, f)
    


def get_model(config, args, dataset):
    """
    Selects, initializes and returns a model instance that is to be trained
    :param args: object from the argument parser
    :param dataset: the dataset that the model will be trained on
    :return: an instantiated model for future training/evaluation
    """
    if args.model == 'ARL':
        model = ARL(config=config, # for hparam tuning
                    num_features=dataset.dimensionality,
                    pretrain_steps=args.pretrain_steps,
                    prim_hidden=args.prim_hidden, 
                    adv_hidden=args.adv_hidden, 
                    #prim_lr=args.prim_lr, # deprecated
                    #adv_lr=args.adv_lr, # deprecated
                    optimizer=OPT_BY_NAME[args.opt],
                    opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})

    elif args.model == 'DRO':
        model = DRO(config=config, # for hparam tuning
                    num_features=dataset.dimensionality,
                    hidden_units=args.prim_hidden,
                    #lr=args.prim_lr, # deprecated
                    #eta=args.eta, # deprecated
                    k=args.k,
                    optimizer=OPT_BY_NAME[args.opt],
                    opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})
        args.pretrain_steps = 0  # NO PRETRAINING

    elif args.model == 'IPW':
        model = IPW(config=config, # for hparam tuning
                    num_features=dataset.dimensionality,
                    hidden_units=args.prim_hidden,
                    #lr=args.prim_lr, # deprecated
                    optimizer=OPT_BY_NAME[args.opt],
                    group_probs=dataset.group_probs,
                    sensitive_label=args.sensitive_label,
                    opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})
        args.pretrain_steps = 0  # NO PRETRAINING

    elif args.model == 'baseline':
        model = BaselineModel(config=config, # for hparam tuning
                              num_features=dataset.dimensionality,
                              hidden_units=args.prim_hidden,
                              #lr=args.prim_lr, # deprecated
                              optimizer=OPT_BY_NAME[args.opt],
                              opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})
        args.pretrain_steps = 0  # NO PRETRAINING

    # if Tensorflow mode is active, we use the TF default initialization,
    # which means Xavier/Glorot uniform (with gain 1) for the weights
    # and 0 bias
    if args.tf_mode:
        def init_weights(layer):
            if type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        model.apply(init_weights)

    return model



def run_folds(config, args, dataset, fold_indices, version=None):
    """
    Function to run kfold cross validation for a given set of parameters
    :param args: object from the argument parser
    :dataset: dataset object containing all training examples
    :fold_indices: list containing tuples of train and val indices
    :version: used to group runs from a single grid search into the same directory
    :return: Results from testing the model
    """
    '''
    # Create datasets
    dummy_train_dataset = Dataset(args.dataset)

    # perform n-fold crossvalidation
    kf = KFold(n_splits=args.num_folds)
    '''

    print(f'Starting run with seed {args.seed} - lr {config["lr"]} - bs {config["batch_size"]}')
    
    fold_nbr = 0
    aucs = []
    for train_idcs, val_idcs in fold_indices:
        fold_nbr += 1

        # create datasets for fold
        train_dataset = CustomSubset(dataset, train_idcs)
        val_dataset = CustomSubset(dataset, val_idcs)

        # train model
        model = train(config, args, train_dataset=train_dataset, val_dataset=val_dataset, version=args.version, fold_nbr=fold_nbr)

        # Evaluate on val set to get an estimate of performance
        scores = torch.sigmoid(model(val_dataset.features)) # suspect of this. Does it work with gpu? doesn't seem to throw an error
        aucs.append(auroc(scores, val_dataset.labels).item())

    mean_auc = np.mean(aucs)
    print(f'Finished run with seed {args.seed} - lr {config["lr"]} - bs {config["batch_size"]} - mean val auc: {mean_auc}')

    tune.report(auc=mean_auc)

    return mean_auc



def train(config, args, train_dataset=None, val_dataset=None, test_dataset=None, version=str(int(time())), fold_nbr=None):            
    
     # create logdir
    logdir = args.log_dir if args.grid_search else os.path.join(args.log_dir, args.dataset, args.model)
    os.makedirs(logdir, exist_ok=True)

    # create fold loaders and callbacks
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    #callbacks = [Logger(train_dataset, 'training', batch_size=config['batch_size'])]
    callbacks = []

    if val_dataset is not None:
        callbacks.append(Logger(val_dataset, 'validation', batch_size=args.eval_batch_size))
        callbacks.append(EarlyStopping(
            monitor='validation/micro_avg_auc',
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode='max'
        ))
    
    if test_dataset is not None:
        callbacks.append(Logger(test_dataset, 'test', batch_size=args.eval_batch_size))
        
    # Select model and instantiate
    model = get_model(config, args, train_dataset)
        
    # create logger
    logger = TensorBoardLogger(
        save_dir='./',
        name=logdir,
        version=(f'version_{version}/seed_{args.seed}/lr_{config["lr"]}_bs_{config["batch_size"]}' if not args.grid_search else '')
                + (f'fold_{fold_nbr}' if fold_nbr is not None else '')
    )
    
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(logger=logger,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, dirpath=logger.log_dir),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_steps=args.train_steps + args.pretrain_steps,
                         callbacks=callbacks,
                         gradient_clip_val=1 if args.model=='DRO' else 0,
                         progress_bar_refresh_rate=1 if args.p_bar else 0,
                         weights_summary=None, # supress model summary
                         #profiler='simple',
                         # fast_dev_run=True # FOR DEBUGGING, SET TO FALSE FOR REAL TRAINING
                         )
    
    # Training
    fit_time = time()
    trainer.fit(model, train_loader)
    print(f'time to fit was {time()-fit_time}')

    return model



if __name__ == '__main__':
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO', 'IPW'], required='True')
    parser.add_argument('--prim_hidden', default=[64, 32], help='Number of hidden units in primary network')
    parser.add_argument('--adv_hidden', default=[32], help='Number of hidden units in adversarial network')
    parser.add_argument('--eta', default=0.7, type=float, help='Threshold for single losses that contribute to learning objective')
    parser.add_argument('--k', default=2.0, type=float, help='Exponent to upweight high losses')

    # Single run settings
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--prim_lr', default=0.1, type=float, help='Learning rate for primary network')
    #parser.add_argument('--adv_lr', default=0.1, type=float, help='Learning rate for adversarial network') # deprecated
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--seed_run', action='store_true', help='Whether this is part of a run with multiple seeds')
    parser.add_argument('--seed_run_version', default=0, type=int, help='Version of the run with multiple seeds')
    
    # More general train settings
    parser.add_argument('--pretrain_steps', default=250, type=int)
    parser.add_argument('--test_steps', default=5, type=int) # deprecated
    parser.add_argument('--opt', choices=['Adagrad'], default="Adagrad", help='Name of optimizer')
    parser.add_argument('--log_dir', default='training_logs', type=str)
    parser.add_argument('--p_bar', action='store_true', help='Whether to use progressbar')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of crossvalidation folds')
    parser.add_argument('--no_grid_search', action='store_false', default=True, dest="grid_search", help='Whether to optimize batch size and lr via gridsearch')
    #parser.add_argument('--nbr_seeds', default=2, type=int, help='Number of independent training runs') # TODO: not implemented yet
    parser.add_argument('--eval_batch_size', default=512, type=int, help='batch size for AUC computation, should be as large as possible')
    parser.add_argument('--tf_mode', action='store_true', default=False, help='Use tensorflow rather than PyTorch defaults where possible. Only supports AdaGrad optimizer.')
    
    # Dataset settings
    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS'], required='True')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers that are used in dataloader')
    parser.add_argument('--disable_warnings', action='store_true', help='Whether to disable warnings about mean and std in the dataset')
    parser.add_argument('--sensitive_label', default=False, type=bool, help='If True, target label will be included in list of sensitive columns')

    # ray settings
    parser.add_argument('--num_cpus', default=1, type=int, help='Number of CPUs used for each trial')
    parser.add_argument('--num_gpus', default=0.25, type=float, help='Number of GPUs used for each trial')

    args = parser.parse_args()

    # run main loop
    main(args)
    
    