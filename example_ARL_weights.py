from typing import Dict, Type, Optional, Any, List, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import TabularDataset, CustomSubset, FairnessDataset, EMNISTDataset
from arl import ARL
from dro import DRO
from ipw import IPW
from baseline_model import BaselineModel
from metrics import Logger, get_all_auc_scores

import argparse
import os

import numpy as np # type: ignore
import pandas as pd # type: ignore
from time import time
import itertools
import json
import warnings

import matplotlib.pyplot as plt

from ray import tune  # type: ignore

from sklearn.model_selection import KFold # type: ignore
from sklearn.neighbors import KernelDensity

print(f'Cuda available? {torch.cuda.is_available()}')


# dict to access optimizers by name, if we need to use different opts.
OPT_BY_NAME: Dict[str, Type[torch.optim.Optimizer]] = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam
}

# obscure bug fix to get ray + slurm + ptl to cooperate
os.environ["SLURM_JOB_NAME"] = "bash"


def main(args: argparse.Namespace):
    """Executes a full grid search (optional) and a single training run.
        
    Grid search: Defines the hyperparameter search space and runs kfold cross 
    validation of the model type as given by the argument parser object for all 
    hyperparameter combinations from the search space. Saves results and best-
    performing hyperparameters with respect to the micro-average AUC.
        
    Single run: Executes a single training run with either the hyperparameters
    found in the grid search or the hyperparameters as given in the argument
    parser object. Evaluates various metrics (AUC, accuracy) on the test dataset
    and saves the scores.   
    
    Args:
        args: Object from the argument parser that defines various settings of
            the model, dataset and training.
    """
    
    # set run version
    args.version = str(int(time()))

    # set model
    args.model = 'ARL'
    
    # seed
    pl.seed_everything(args.seed)
    np.random.seed(args.seed)

    # create datasets
    if args.dataset == 'EMNIST':
        dataset: FairnessDataset = EMNISTDataset()
        test_dataset: FairnessDataset = EMNISTDataset(test=True)
    else:
        dataset = TabularDataset(args.dataset, disable_warnings=args.disable_warnings)
        test_dataset = TabularDataset(args.dataset, test=True, disable_warnings=args.disable_warnings)
    
    # init config dictionary
    config: Dict[str, Any] = {}
    
    
    # set hparams for single run
    config['lr'] = args.prim_lr
    config['batch_size'] = args.batch_size
    
    path = f'./{args.log_dir}/{args.dataset}/ARL_example_weights/version_{args.version}'
    
    print(f'creating dir {path}')
    os.makedirs(path, exist_ok=True)

    # set log_dir
    args.log_dir = path

    # create val and train set
    permuted_idcs = np.random.permutation(np.arange(0, len(dataset)))
    train_idcs, val_idcs = permuted_idcs[:int(0.9*len(permuted_idcs))], permuted_idcs[int(0.9*len(permuted_idcs)):] 
    train_dataset, val_dataset = CustomSubset(dataset, train_idcs), CustomSubset(dataset, val_idcs)
    
    # single training run
    model: pl.LightningModule = train(config, args, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
    
    # compute final test scores
    dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    auc_scores: Dict[str, float] = get_all_auc_scores(model, dataloader, test_dataset.minority)
        
    # print results
    print(f'AUC Results = {auc_scores}')

    # eval adversary scores
    print(f'Evaluating adversary scores on test set')
    lambdas, predictions, true_labels, memberships = model.get_lambda(dataloader)

    # detach and reshape
    lambdas = lambdas.detach().numpy().reshape(-1,1)
    predictions = predictions.detach().numpy().reshape(-1,1)
    true_labels = true_labels.detach().numpy().reshape(-1,1)
    memberships = memberships.detach().numpy().reshape(-1,1)

    # sort in ascending order of lambda
    sort_idcs = np.argsort(lambdas, axis=0)
    lambdas = lambdas[sort_idcs]
    predictions = predictions[sort_idcs]
    true_labels = true_labels[sort_idcs]
    memberships = memberships[sort_idcs]

    if not args.notebook:
        print(f'Saving results to {path}')
        np.save(os.path.join(path, 'lambdas'), lambdas)
        np.save(os.path.join(path, 'predictions'), predictions)
        np.save(os.path.join(path, 'true_labels'), true_labels)
        np.save(os.path.join(path, 'memberships'), memberships)

        print('Plotting adversary scores..')
    # init KD
    kde = KernelDensity(bandwidth=0.3)

    # get idx2val dict from test dataset
    # e.g. {0: ('Other', 'Other'), 1: ('Other', 'Female'), 2: ('Black', 'Other'), 3: ('Black', 'Female')}
    idx2val = test_dataset.protected_index2value
    race_dict = {'Black':'Black', 'Other':'White'}
    sex_dict = {'Female':'Female', 'Other':'Male'}

    #
    score_ticks = np.linspace(0,5,100).reshape((100,1))

    if not args.notebook:
        # plot 0 - 0
        plt.figure()
        for i in idx2val:
            combi = idx2val[i]
            lam = lambdas[(predictions == 0) & (true_labels == 0) & (memberships == i)][:, np.newaxis]
            kde_00 = kde.fit(lam)
            density_00 = np.exp(kde_00.score_samples(score_ticks))
            plt.plot(score_ticks[:,0], density_00, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
        plt.legend()
        plt.savefig(os.path.join(path, 'lambda_0_0.pdf'))

        # plot 0 - 1
        plt.figure()
        for i in idx2val:
            combi = idx2val[i]
            lam = lambdas[(predictions == 1) & (true_labels == 0) & (memberships == i)][:, np.newaxis]
            kde_01 = kde.fit(lam)
            density_01 = np.exp(kde_01.score_samples(score_ticks))
            plt.plot(score_ticks, density_01, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
        plt.legend()
        plt.savefig(os.path.join(path, 'lambda_0_1.pdf'))

        # plot 1 - 0
        plt.figure()
        for i in idx2val:
            combi = idx2val[i]
            lam = lambdas[(predictions == 0) & (true_labels == 1) & (memberships == i)][:, np.newaxis]
            kde_10 = kde.fit(lam)
            density_10 = np.exp(kde_10.score_samples(score_ticks))
            plt.plot(score_ticks, density_10, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
        plt.legend()
        plt.savefig(os.path.join(path, 'lambda_1_0.pdf'))

        # plot 1 - 1
        plt.figure()
        for i in idx2val:
            combi = idx2val[i]
            lam = lambdas[(predictions == 1) & (true_labels == 1) & (memberships == i)][:, np.newaxis]
            kde_11 = kde.fit(lam)
            density_11 = np.exp(kde_11.score_samples(score_ticks))
            plt.plot(score_ticks, density_11, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
        plt.legend()
        plt.savefig(os.path.join(path, 'lambda_1_1.pdf'))

    # combined plot
    f, axes = plt.subplots(2,2,gridspec_kw={'hspace':.5})
    axes = axes.flatten()
    
    plt_settings = [(0,0), (0,1), (1,0), (1,1)]
    titles = ['no-error; class 0', 'error; class 0', 'error; class 1', 'no-error; class 1']
    for i in range(len(plt_settings)):
        for idx in idx2val:
            combi = idx2val[idx]
            lam = lambdas[(predictions == plt_settings[i][1]) & (true_labels == plt_settings[i][0]) & (memberships == idx)][:, np.newaxis]
            kde_combined = kde.fit(lam)
            density_combined = np.exp(kde_combined.score_samples(score_ticks))
            axes[i].plot(score_ticks, density_combined, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
            if i == 0 or i == 2:
                axes[i].set_ylabel('density')
            if i == 2 or i == 3:
                axes[i].set_xlabel(r'test weight $\lambda$')
            axes[i].set_title(titles[i])
            if i == 0:
                axes[i].legend()
    if args.notebook:
        plt.show()
    else:
        f.savefig(os.path.join(path, 'lambdas_combined.pdf'))
    


    # save results
    with open(os.path.join(path, 'auc_scores.json'),'w') as f:
        json.dump(auc_scores, f)
    

def get_model(config: Dict[str, Any], args: argparse.Namespace, dataset: FairnessDataset) -> pl.LightningModule:
    """Selects and inits a model instance for training.
    
    Args:
        config: Dict with hyperparameters (learning rate, batch size, eta).
        args: Object from the argument parser that defines various settings of
            the model, dataset and training.
        dataset: Dataset instance that will be used for training.
    
    Returns:
        An instantiated model; one of the following:
                
        Model based on Adversarially Reweighted Learning (ARL).
        Model based on Distributionally Robust Optimization (DRO).
        Model based on Inverse Probability Weighting (IPW).
        Baseline model; simple fully-connected or convolutional (TODO) network.
    """
    
    model: pl.LightningModule
    
    model = ARL(config=config, # for hparam tuning
                input_shape=dataset.dimensionality,
                pretrain_steps=args.pretrain_steps,
                prim_hidden=args.prim_hidden, 
                adv_hidden=args.adv_hidden, 
                optimizer=OPT_BY_NAME[args.opt],
                dataset_type=args.dataset_type,
                adv_input=set(args.adv_input),
                num_groups=len(dataset.protected_index2value),
                opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})

    if args.tf_mode:
        def init_weights(layer):
            if type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        model.apply(init_weights)

    return model



def train(config: Dict[str, Any],
          args: argparse.Namespace,
          train_dataset: FairnessDataset,
          val_dataset: Optional[FairnessDataset]=None,
          test_dataset: Optional[FairnessDataset]=None,
          version=str(int(time()))) -> pl.LightningModule:
    """Single training run on a given dataset.
    
    Inits a model and optimizes its parameters on the given training dataset  
    with a given set of hyperparameters. Logs various metrics and stops the 
    training when the micro-average AUC on the validation set stops improving.
    
    Args:
        config: Dict with hyperparameters (learning rate, batch size, eta).
        args: Object from the argument parser that defines various settings 
            of the model, dataset and training.
        train_dataset: Dataset instance to use for training.
        val_dataset: Optional; dataset instance to use for validation.
        test_dataset: Optional; dataset instance to use for testing.
        version: Version used for the logging directory.
    
    Returns:
        Model with the highest micro-average AUC on the validation set during 
        the training run.
            
    Raises:
        AssertionError: If no model checkpoint callback exists.
    """
    
    # create logdir if necessary
    logdir: str = args.log_dir
    os.makedirs(logdir, exist_ok=True)

    # create fold loaders and callbacks
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    callbacks: List[pl.callbacks.Callback] = []

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
    model: pl.LightningModule = get_model(config, args, train_dataset)
        
    # create logger
    logger_version = f'seed_{args.seed}'
    
    logger = TensorBoardLogger(
        save_dir='./',
        name=logdir,
        version=logger_version
    )

    # create checkpoint
    checkpoint = ModelCheckpoint(save_weights_only=True,
                                dirpath=logger.log_dir, 
                                mode='max', 
                                verbose=False,
                                monitor='validation/micro_avg_auc')
    callbacks.append(checkpoint)
    
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(logger=logger,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_steps=args.train_steps + args.pretrain_steps,
                         callbacks=callbacks,
                         gradient_clip_val=1 if args.model=='DRO' else 0,
                         progress_bar_refresh_rate=1 if args.p_bar else 0,
                         )
    
    # Training
    fit_time = time()
    if val_dataset is not None:
        trainer.fit(model, train_loader, val_dataloaders=DataLoader(val_dataset, batch_size=args.eval_batch_size))
    else:
        trainer.fit(model, train_loader)
    print(f'time to fit was {time()-fit_time}')

    # necessary to make the type checker happy and since this is only run once,
    # runtime is not an issue
    assert trainer.checkpoint_callback is not None

    # Load best checkpoint after training
    if args.model == 'ARL' and args.dataset == 'EMNIST':
        model = ARL.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
    elif args.model == 'ARL':
        model = ARL.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
    return model



if __name__ == '__main__':
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--prim_hidden', nargs='*', type=int, default=[64, 32], help='Number of hidden units in primary network')
    parser.add_argument('--adv_hidden', nargs='*', type=int, default=[], help='Number of hidden units in adversarial network')
    parser.add_argument('--adv_input', nargs='+', default=['X', 'Y'], help='Inputs to use for the adversary. Any combination of X (features), Y (labels) and S (protected group memberships)')

    # Single run settings
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--prim_lr', default=0.1, type=float, help='Learning rate for primary network')
    parser.add_argument('--seed', default=0, type=int)
    
    # More general train settings
    parser.add_argument('--pretrain_steps', default=250, type=int)
    parser.add_argument('--opt', choices=OPT_BY_NAME.keys(), default="Adagrad", help='Name of optimizer')
    parser.add_argument('--log_dir', default='training_logs', type=str)
    parser.add_argument('--p_bar', action='store_true', help='Whether to use progressbar')
    parser.add_argument('--eval_batch_size', default=512, type=int, help='Batch size for evaluation. No effect on training or results, set as large as memory allows to maximize performance')
    parser.add_argument('--tf_mode', action='store_true', default=False, help='Use tensorflow rather than PyTorch defaults where possible. Only supports AdaGrad optimizer.')
    
    # Dataset settings
    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS', 'EMNIST'], default='Adult')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers that are used in dataloader')
    parser.add_argument('--disable_warnings', action='store_true', help='Whether to disable warnings about mean and std in the dataset')

    # ray settings
    parser.add_argument('--num_cpus', default=1, type=int, help='Number of CPUs used for each trial')
    parser.add_argument('--num_gpus', default=0.25, type=float, help='Number of GPUs used for each trial')
    parser.add_argument('--notebook', default=False, action='store_true', help='Use notebook mode (which won\'t save anything and create fewer plots')

    args: argparse.Namespace = parser.parse_args()

    args.working_dir = os.getcwd()

    # run main loop
    main(args)
    
