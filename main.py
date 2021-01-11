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
    """
    Runs grid search for hyperparameters
    :param args: object from the argument parser
    :return: not implemented
    """
    # set run version
    version = str(int(time()))

    # specify search space 
    # TODO: pull this outside this function for more flexible search space?
    lr_list = [0.001, 0.01, 0.1, 1, 2, 5]
    batch_size_list = [32, 64, 128, 256, 512]

    
    # perform n-fold crossvalidation
    kf = KFold(n_splits=args.num_folds)

    # create datasets
    dataset = CustomDataset(args.dataset)
    fold_indices = list(kf.split(dataset))
    
    # find best hparams
    best_mean_auc = 0
    for lr, bs in itertools.product(lr_list, batch_size_list):
        args.prim_lr, args.adv_lr = lr, lr
        args.batch_size = bs
        mean_auc = run_folds(args, dataset=dataset, fold_indices=fold_indices, version=version+'_folds')
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_bs = bs
            best_lr = lr

    # Test model
    args.prim_lr, args.adv_lr, args.batch_size = best_lr, best_lr, best_bs
    full_train_test(args, version=version + '_test')

    # save parameters
    best_params = {'learning_rate': best_lr, 'batch_size': best_bs}
    # write best params to file
    with open(os.path.join(args.log_dir, args.dataset, args.model, f'version_{version}_test', 'best_params.json'), 'w') as f:
        json.dump(best_params, f)

    print(f'Best hyperparameters: {best_params}')


def get_model(args, dataset):
    """
    Selects, initializes and returns a model instance that is to be trained
    :param args: object from the argument parser
    :param dataset: the dataset that the model will be trained on
    :return: an instantiated model for future training/evaluation
    """
    if args.model == 'ARL':
        model = ARL(num_features=dataset.dimensionality,
                    pretrain_steps=args.pretrain_steps,
                    prim_hidden=args.prim_hidden,
                    adv_hidden=args.adv_hidden,
                    prim_lr=args.prim_lr,
                    adv_lr=args.adv_lr,
                    optimizer=OPT_BY_NAME[args.opt],
                    opt_kwargs={})

    elif args.model == 'DRO':
        model = DRO(num_features=dataset.dimensionality,
                    hidden_units=args.prim_hidden,
                    lr=args.prim_lr,
                    eta=args.eta,
                    k=args.k,
                    optimizer=OPT_BY_NAME[args.opt],
                    opt_kwargs={})
        args.pretrain_steps = 0  # NO PRETRAINING

    elif args.model == 'IPW':
        model = IPW(num_features=dataset.dimensionality,
                    hidden_units=args.prim_hidden,
                    lr=args.prim_lr,
                    optimizer=OPT_BY_NAME[args.opt],
                    group_probs=dataset.group_probs,
                    opt_kwargs={})
        args.pretrain_steps = 0  # NO PRETRAINING

    elif args.model == 'baseline':
        model = BaselineModel(num_features=dataset.dimensionality,
                              hidden_units=args.prim_hidden,
                              lr=args.prim_lr,
                              optimizer=OPT_BY_NAME[args.opt],
                              opt_kwargs={})
        args.pretrain_steps = 0  # NO PRETRAINING

    return model

    
def run_folds(args, dataset, fold_indices, version=None):
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
    fold_nbr = 0
    aucs = []
    for train_idcs, val_idcs in fold_indices:
        fold_nbr += 1
        
        # create datasets for fold
        train_dataset = CustomSubset(dataset, train_idcs)
        val_dataset = CustomSubset(dataset, val_idcs)

        # train model
        model = train(args, train_dataset=train_dataset, val_dataset=val_dataset, version=version, fold_nbr=fold_nbr)

        # Evaluate on val set to get an estimate of performance
        scores = torch.sigmoid(model(val_dataset.features))
        aucs.append(auroc(scores, val_dataset.labels).item())

    mean_auc = np.mean(aucs)
    print(f'mean val auc: {mean_auc}')

    return mean_auc


def train(args, train_dataset=None, val_dataset=None, test_dataset=None, version=str(int(time())), fold_nbr=None):
    """
    Function to train a model
    :param args: object from the argument parser
    :param train_dataset: dataset instance for training
    :param val_dataset: optional dataset instance if validation shall be done
    :param test_dataset: optional dataset instance if testing shall be done
    :param version: used to group runs from a single grid search into the same directory
    :param fold_nbr: number of the fold if kfold cross validation is used
    :return: The trained model instance
    """

    # create logdir
    logdir = os.path.join(args.log_dir, args.dataset, args.model)
    os.makedirs(logdir, exist_ok=True)

    # create fold loaders and callbacks
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    callbacks = [Logger(train_dataset, 'training')]

    if val_dataset is not None:
        callbacks.append(Logger(val_dataset, 'validation'))
        callbacks.append(EarlyStopping(
            monitor='validation/micro_avg_auc',
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode='max'
        ))

    if test_dataset is not None:
        callbacks.append(Logger(test_dataset, 'test'))

    # Select model and instantiate
    model = get_model(args, train_dataset)

    # create logger
    logger = TensorBoardLogger(
        save_dir='./',
        name=logdir,
        version=f'version_{version}/lr_{args.prim_lr}_bs_{args.batch_size}'
                + (f'/fold_{fold_nbr}' if fold_nbr is not None else '')
    )

    # raise ValueError
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(logger=logger,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, dirpath=logger.log_dir),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_steps=args.train_steps + args.pretrain_steps,
                         callbacks=callbacks,
                         progress_bar_refresh_rate=1 if args.p_bar else 0
                         # fast_dev_run=True # FOR DEBUGGING, SET TO FALSE FOR REAL TRAINING
                         )

    # Training
    trainer.fit(model, train_loader)

    return model


def full_train_test(args, version=str(int(time()))):
    """
    Trains a model on the complete training dataset and evaluates on test set
    :param args: object from the argument parser
    :param version: used to group runs from a single grid search into the same directory
    :return: not implemented
    """
    # create datasets
    train_dataset = CustomDataset(args.dataset, disable_warnings=args.disable_warnings)
    test_dataset = CustomDataset(args.dataset, test=True, disable_warnings=args.disable_warnings)

    # run training and testing
    train(args, train_dataset=train_dataset, test_dataset=test_dataset, version=version)



if __name__ == "__main__":
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO', 'IPW'], required=True)
    parser.add_argument('--prim_hidden', default=[64, 32], help='Number of hidden units in primary network')
    parser.add_argument('--adv_hidden', default=[32], help='Number of hidden units in adversarial network')
    parser.add_argument('--eta', default=0.2, type=float, help='Threshold for single losses that contribute to learning objective')
    parser.add_argument('--k', default=2.0, type=float, help='Norm of the loss function')

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

    # Seed for reproducability
    pl.seed_everything(args.seed)

    if args.grid_search:
        grid_search(args)
    else:
        # run training loop
        full_train_test(args)
