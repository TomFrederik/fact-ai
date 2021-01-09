import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import Dataset
from arl import ARL
from baseline_model import BaselineModel
from metrics import Logger

import argparse
import os

import numpy as np
from time import time
import itertools
import json

from sklearn.model_selection import KFold

# dict to access optimizers by name, if we need to use different opts.
OPT_BY_NAME = {'Adagrad': torch.optim.Adagrad}


def grid_search(args):
    '''
    runs gridsearch
    '''
    # set run version
    version = str(int(time()))

    # specify search space 
    # TODO: pull this outside this function for more flexible search space?
    lr_list = [0.001, 0.01, 0.1, 1, 2, 5]
    batch_size_list = [32, 64, 128, 256, 512]

    # find best hparams
    best_mean_auc = 0
    
    for lr, bs in itertools.product(lr_list, batch_size_list):
        args.lr = lr
        args.batch_size = bs
        mean_auc = train(args, version)
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_bs = bs
            best_lr = lr
    
    best_params = {'learning_rate': best_lr, 'batch_size': best_bs}
    # write best params to file
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
    
    print(f'Best hyperparameters: {best_params}')
    
def train(args, version=None):
    """
    Function to train a model
    :param args: object from the argument parser
    :version: used to group runs from a single grid search into the same directory
    :return: Results from testing the model
    """

    # create logdir
    logdir = os.path.join(args.log_dir, args.dataset, args.model)
    os.makedirs(logdir, exist_ok=True)
    if version is None:
        version = str(int(time()))

    # Seed for reproducability
    pl.seed_everything(args.seed)

    # Create datasets
    dummy_train_dataset = Dataset(args.dataset)

    test_dataset = Dataset(args.dataset, test=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up callbacks which stay constant
    test_callback = Logger(test_dataset, 'test')
    
    # perform n-fold crossvalidation
    kf = KFold(n_splits=args.num_folds)
    fold_nbr = 0
    aucs = []
    for train_idcs, val_idcs in kf.split(dummy_train_dataset):
        fold_nbr += 1

        # create fold datasets, loaders and callbacks
        train_dataset = Dataset(args.dataset, idcs=train_idcs, disable_warnings=args.disable_warnings)
        val_dataset = Dataset(args.dataset, idcs=val_idcs, disable_warnings=args.disable_warnings)

        train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

        validation_loader = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
        
        train_callback = Logger(train_dataset, 'training')
        val_callback = Logger(val_dataset, 'validation')
        early_stop_callback = EarlyStopping(
            monitor='validation/micro_avg_auc',
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode='max'
        )
    

        # Select model and instantiate
        if args.model == 'ARL':
            model = ARL(num_features=train_dataset.dimensionality,
                        pretrain_steps=args.pretrain_steps,
                        prim_hidden=args.prim_hidden,
                        adv_hidden=args.adv_hidden,
                        prim_lr=args.prim_lr,
                        adv_lr=args.adv_lr,
                        optimizer=OPT_BY_NAME[args.opt],
                        opt_kwargs={})

        elif args.model == 'DRO':
            raise NotImplementedError
        
        elif args.model == 'baseline':
            model = BaselineModel(num_features=train_dataset.dimensionality,
                                hidden_units=args.prim_hidden,
                                lr=args.prim_lr,
                                optimizer=OPT_BY_NAME[args.opt],
                                opt_kwargs={})
            args.pretrain_steps = 0 # NO PRETRAINING

        # create logger
        logger = TensorBoardLogger(save_dir='./', name=logdir, version=f'version_{version}/lr_{args.lr}_bs_{args.batch_size}/fold_{fold_nbr}')

        #raise ValueError
        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(logger=logger,
                            checkpoint_callback=ModelCheckpoint(save_weights_only=True, dirpath=logger.log_dir),
                            gpus=1 if torch.cuda.is_available() else 0,
                            max_steps=args.train_steps+args.pretrain_steps,
                            callbacks=[train_callback, val_callback, early_stop_callback], # test callback?
                            progress_bar_refresh_rate=1 if args.p_bar else 0
                            #fast_dev_run=True # FOR DEBUGGING, SET TO FALSE FOR REAL TRAINING
                            )

        # Training
        trainer.fit(model, train_loader)

        # Evaluate on val set to get an estimate of performance
        scores = torch.sigmoid(model(val_dataset.features))
        aucs.append(auroc(scores, val_dataset.labels).item())
    
    mean_auc = np.mean(aucs)
    print(f'mean val auc: {mean_auc}')

    # Testing
    #model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    #test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)

    #return mean_auc
    return mean_auc

if __name__ == "__main__":
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO'], required=True)
    parser.add_argument('--prim_hidden', default=[64, 32], help='Number of hidden units in primary network')
    parser.add_argument('--adv_hidden', default=[32], help='Number of hidden units in adversarial network')

    # Training settings
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_steps', default=20, type=int)
    parser.add_argument('--pretrain_steps', default=250, type=int)
    parser.add_argument('--test_steps', default=5, type=int)
    parser.add_argument('--opt', choices=['Adagrad'], default="Adagrad", help='Name of optimizer')
    parser.add_argument('--prim_lr', default=0.1, type=float, help='Learning rate for primary network')
    parser.add_argument('--adv_lr', default=0.1, type=float, help='Learning rate for adversarial network')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log_dir', default='training_logs', type=str)
    parser.add_argument('--p_bar', action='store_true', help='Whether to use progressbar')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of crossvalidation folds')
    parser.add_argument('--grid_search', default=False, type=bool, help='Whether to optimize batch size and lr via gridsearch')

    # Dataset settings
    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS'], required=True)
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers that are used in dataloader')
    parser.add_argument('--disable_warnings', default=False, type=bool, help='Whether to disable warnings about mean and std in the dataset')

    args = parser.parse_args()

    if args.grid_search:
        grid_search(args)
    else:
        # run training loop
        train(args)
