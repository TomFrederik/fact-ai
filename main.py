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
from typing import Tuple
import itertools
import json
import warnings

from ray import tune  # type: ignore

from sklearn.model_selection import KFold # type: ignore

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

    # make a copy so we don't change the args object
    args = argparse.Namespace(**vars(args))
    args.dataset_type = 'image' if args.dataset in ['EMNIST_35', 'EMNIST_10'] else 'tabular'
    
    if args.version is None:
        # set run version
        args.version = str(int(time()))
        
    # seed
    pl.seed_everything(args.seed)
    np.random.seed(args.seed)

    # create datasets
    if args.dataset == 'EMNIST_35':
        dataset: FairnessDataset = EMNISTDataset()
        test_dataset: FairnessDataset = EMNISTDataset(test=True)
    elif args.dataset == 'EMNIST_10':
        dataset: FairnessDataset = EMNISTDataset(imb=True)
        test_dataset: FairnessDataset = EMNISTDataset(imb=True, test=True)
    else:
        dataset = TabularDataset(args.dataset, sensitive_label=args.sensitive_label, disable_warnings=args.disable_warnings)
        test_dataset = TabularDataset(args.dataset, sensitive_label=args.sensitive_label, test=True, disable_warnings=args.disable_warnings)
    
    # init config dictionary
    config: Dict[str, Any] = {}
    
    if args.grid_search and args.dataset_type == 'tabular':
        # specify search space 
        lr_list: List[float] = [0.001, 0.01, 0.1, 1, 2, 5]
        batch_size_list: List[int] = [32, 64, 128, 256, 512]
        eta_list: List[float] = [0.0] # dummy entry for non-DRO experiments
        
        if args.model == 'DRO':
            eta_list = [0.5, 0.6, 0.7, 0.8, 0.9]    
    
        # configurations for hparam tuning
        config = {
            'lr': tune.grid_search(lr_list),
            'sec_lr': tune.grid_search(lr_list),
            'batch_size': tune.grid_search(batch_size_list),
            'eta': tune.grid_search(eta_list)
            }
      
        # perform n-fold crossvalidation
        kf = KFold(n_splits=args.num_folds)          
        fold_indices: List[Tuple[np.ndarray, np.ndarray]] = list(kf.split(dataset))
        
        # set path for logging
        if args.model == 'IPW':
            if args.sensitive_label:
                path = f'grid_search/IPW(S+Y)_{args.dataset}_version_{args.version}'
            else:
                path = f'grid_search/IPW(S)_{args.dataset}_version_{args.version}'
        else:
            path = f'grid_search/{args.model}_{args.dataset}_version_{args.version}'
        
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
        df: pd.DataFrame = analysis.results_df
        df.to_csv(os.path.join(path, 'results.csv'), index=False)
        
        # set hparams for final run
        config['lr'] = analysis.best_config['lr']
        config['sec_lr'] = analysis.best_config['sec_lr']
        config['batch_size'] = analysis.best_config['batch_size']
        config['eta'] = analysis.best_config['eta']

    elif args.grid_search and args.dataset_type == 'image':
        lr_list: List[float] = [0.001, 0.01, 0.1, 1, 2, 5]
        batch_size_list: List[int] = [32, 64, 128, 256, 512]

        best_score, best_lr, best_batch_size = -np.infty, None, None

        # set path
        path = f'grid_search/{args.model}_{args.dataset}_version_{args.version}'

        # create val and train set
        permuted_idcs = np.random.permutation(np.arange(0, len(dataset)))
        train_idcs, val_idcs = permuted_idcs[:int(0.9 * len(permuted_idcs))], permuted_idcs[int(0.9 * len(permuted_idcs)):]
        train_dataset, val_dataset = CustomSubset(dataset, train_idcs), CustomSubset(dataset, val_idcs)

        for lr, batch_size in itertools.product(lr_list, batch_size_list):
            config = {
                'lr': lr,
                'batch_size': batch_size,
            }

            # set logdir
            args.log_dir = os.path.join(path, f"batch_size={config['batch_size']}_lr={config['lr']}")

            # training run
            t: Tuple[pl.LightningModule, pl.Trainer] = train(config, args, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
            model, trainer = t
            if trainer.checkpoint_callback.best_model_score > best_score:
                best_score = trainer.checkpoint_callback.best_model_score
                best_lr = config['lr']
                best_batch_size = config['batch_size']

        print(f'Best hyperparameters found were: lr = {best_lr}, batch_size = {best_batch_size}')

        # set hparams for final run
        config['lr'] = best_lr
        config['batch_size'] = best_batch_size

    else:
        # set hparams for single run
        config['lr'] = args.prim_lr
        config['sec_lr'] = args.sec_lr if args.sec_lr is not None else args.prim_lr
        config['batch_size'] = args.batch_size
        config['eta'] = args.eta
        
        if args.seed_run:
            if args.model == 'IPW':
                if args.sensitive_label:
                    path = f'./{args.log_dir}/{args.dataset}/IPW(S+Y)/seed_run_version_{args.seed_run_version}/seed_{args.seed}'
                else:
                    path = f'./{args.log_dir}/{args.dataset}/IPW(S)/seed_run_version_{args.seed_run_version}/seed_{args.seed}'
            else:
                path = f'./{args.log_dir}/{args.dataset}/{args.model}/seed_run_version_{args.seed_run_version}/seed_{args.seed}'
        else:
            if args.model == 'IPW':
                if args.sensitive_label:
                    path = f'./{args.log_dir}/{args.dataset}/IPW(S+Y)/version_{args.version}'
                else:
                    path = f'./{args.log_dir}/{args.dataset}/IPW(S)/version_{args.version}'
            else:
                path = f'./{args.log_dir}/{args.dataset}/{args.model}/version_{args.version}'

        print(f'creating dir {path}')
        os.makedirs(path, exist_ok=True)

    # set log_dir
    args.log_dir = path

    # create val and train set
    permuted_idcs = np.random.permutation(np.arange(0, len(dataset)))
    train_idcs, val_idcs = permuted_idcs[:int(0.9*len(permuted_idcs))], permuted_idcs[int(0.9*len(permuted_idcs)):] 
    train_dataset, val_dataset = CustomSubset(dataset, train_idcs), CustomSubset(dataset, val_idcs)
    
    # single training run
    t: Tuple[pl.LightningModule, pl.Trainer] = train(config, args, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
    model, _ = t

    # compute final test scores
    dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)
    auc_scores: Dict[str, float] = get_all_auc_scores(model, dataloader, test_dataset.minority)

    # print results
    print(f'Results = {auc_scores}')
    
    # save results
    with open(os.path.join(path, 'auc_scores.json'),'w') as f:
        json.dump(auc_scores, f)

    if args.grid_search and args.dataset_type != 'image':
        return auc_scores, analysis.best_config
    
    elif args.grid_search and args.dataset_type == 'image':
        return auc_scores, {'lr': best_lr, 'batch_size': best_batch_size}

    else:
        return auc_scores
    

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

    if args.model == 'ARL':
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

    elif args.model == 'ARL_strong':
        model = ARL(config=config, # for hparam tuning
                    input_shape=dataset.dimensionality,
                    pretrain_steps=args.pretrain_steps,
                    prim_hidden=args.prim_hidden, 
                    adv_hidden=args.adv_hidden, 
                    optimizer=OPT_BY_NAME[args.opt],
                    dataset_type=args.dataset_type,
                    adv_input=set(args.adv_input),
                    num_groups=len(dataset.protected_index2value),
                    adv_cnn_strength='strong',
                    opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})
    elif args.model == 'ARL_weak':
        model = ARL(config=config, # for hparam tuning
                    input_shape=dataset.dimensionality,
                    pretrain_steps=args.pretrain_steps,
                    prim_hidden=args.prim_hidden, 
                    adv_hidden=args.adv_hidden, 
                    optimizer=OPT_BY_NAME[args.opt],
                    dataset_type=args.dataset_type,
                    adv_input=set(args.adv_input),
                    num_groups=len(dataset.protected_index2value),
                    adv_cnn_strength='weak',
                    opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})
    elif args.model == 'DRO':
        model = DRO(config=config, # for hparam tuning
                    num_features=dataset.dimensionality,
                    hidden_units=args.prim_hidden,
                    pretrain_steps=args.pretrain_steps,
                    k=args.k,
                    optimizer=OPT_BY_NAME[args.opt],
                    opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})

    elif args.model == 'IPW':
        model = IPW(config=config, # for hparam tuning
                    num_features=dataset.dimensionality,
                    hidden_units=args.prim_hidden,
                    optimizer=OPT_BY_NAME[args.opt],
                    group_probs=dataset.group_probs,
                    sensitive_label=args.sensitive_label,
                    opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})
        args.pretrain_steps = 0  # NO PRETRAINING

    elif args.model == 'baseline':
        model = BaselineModel(config=config, # for hparam tuning
                              num_features=dataset.dimensionality,
                              hidden_units=args.prim_hidden,
                              optimizer=OPT_BY_NAME[args.opt],
                              dataset_type=args.dataset_type,
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



def run_folds(config: Dict[str, Any],
              args: argparse.Namespace,
              dataset: FairnessDataset,
              fold_indices: List[Tuple[np.ndarray, np.ndarray]],
              version: Optional[str]=None) -> float:
    """Runs kfold cross validation on the given dataset.
    
    Executes single training runs on the training set of each fold and evaluates
    the trained model on the validation set of the same fold.
    
    Args:
        config: Dict with hyperparameters (learning rate, batch size, eta).
        args: Object from the argument parser that defines various settings of
            the model, dataset and training.
        dataset: Dataset instance that will be used for cross validation.
        fold_indices: Indices to select training and validation subsets for each
            fold.
        version: Optional; version used for the logging directory.
    
    Returns:
        Mean of the micro-average AUC of the trained models on the validation
        sets of the models' corresponding folds.
    """
    
    print(f'Starting run with seed {args.seed} - lr {config["lr"]} - sec_lr {config["sec_lr"]} - bs {config["batch_size"]}')
    
    fold_nbr = 0
    aucs: List[float] = []
    for train_idcs, val_idcs in fold_indices:
        fold_nbr += 1

        # create datasets for fold
        train_dataset = CustomSubset(dataset, train_idcs)
        val_dataset = CustomSubset(dataset, val_idcs)

        # train model
        t: Tuple[pl.LightningModule, pl.Trainer] = train(config, args, train_dataset=train_dataset,
                                          val_dataset=val_dataset,
                                          version=args.version, fold_nbr=fold_nbr)
        model, _ = t

        # Evaluate on val set to get an estimate of performance
        scores: torch.Tensor = torch.sigmoid(model(val_dataset.features))
        aucs.append(auroc(scores, val_dataset.labels).item())

    mean_auc: float = np.mean(aucs)
    print(f'Finished run with seed {args.seed} - lr {config["lr"]} - sec_lr {config["sec_lr"]} - bs {config["batch_size"]} - mean val auc:'
          f' {mean_auc}')

    tune.report(auc=mean_auc)

    return mean_auc



def train(config: Dict[str, Any],
          args: argparse.Namespace,
          train_dataset: FairnessDataset,
          val_dataset: Optional[FairnessDataset]=None,
          test_dataset: Optional[FairnessDataset]=None,
          version=str(int(time())),
          fold_nbr=None) -> Tuple[pl.LightningModule, pl.Trainer]:
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
        fold_nbr: Optional; used for the logging directory if training run
            is part of kfold cross validation.
    
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
    callbacks.append(Logger(train_dataset, 'train', batch_size=args.eval_batch_size, save_scatter=(args.model in ['ARL', 'ARL_strong', 'ARL_weak'])))

    if val_dataset is not None:
        callbacks.append(Logger(val_dataset, 'validation', batch_size=args.eval_batch_size))
        if not args.no_early_stopping:
            callbacks.append(EarlyStopping(
                monitor='validation/micro_avg_auc', 
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='max'
            ))
    
    if test_dataset is not None:
        callbacks.append(Logger(test_dataset, 'test', batch_size=args.eval_batch_size, save_scatter=(args.model in ['ARL', 'ARL_strong', 'ARL_weak'])))

    # Select model and instantiate
    model: pl.LightningModule = get_model(config, args, train_dataset)
        
    # create logger
    if args.grid_search:
        logger_version = ''
    else:
        logger_version = f'seed_{args.seed}'
    if fold_nbr is not None:
        logger_version += f'./fold_{fold_nbr}'
    
    logger = TensorBoardLogger(
        save_dir='./',
        name=logdir,
        version=logger_version
    )

    if not args.no_early_stopping:
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
        trainer.fit(model, train_loader, val_dataloaders=DataLoader(val_dataset,
                                                                    batch_size=args.eval_batch_size,
                                                                    num_workers=args.num_workers))
    else:
        trainer.fit(model, train_loader)
    print(f'time to fit was {time()-fit_time}')

    if not args.no_early_stopping:
        # necessary to make the type checker happy and since this is only run once,
        # runtime is not an issue
        assert trainer.checkpoint_callback is not None

        # Load best checkpoint after training
        if args.model == 'baseline':
            model = BaselineModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        elif args.model == 'ARL':
            model = ARL.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        elif args.model == 'DRO':
            model = DRO.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        elif args.model == 'IPW':
            model = IPW.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model, trainer



if __name__ == '__main__':
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO', 'IPW', 'ARL_strong', 'ARL_weak'], required=True)
    parser.add_argument('--prim_hidden', nargs='*', type=int, default=[64, 32], help='Number of hidden units in primary network')
    parser.add_argument('--adv_hidden', nargs='*', type=int, default=[], help='Number of hidden units in adversarial network')
    parser.add_argument('--eta', default=0.5, type=float, help='Threshold for single losses that contribute to learning objective')
    parser.add_argument('--k', default=2.0, type=float, help='Exponent to upweight high losses')
    parser.add_argument('--adv_input', nargs='+', default=['X', 'Y'], help='Inputs to use for the adversary. Any combination of X (features), Y (labels) and S (protected group memberships)')

    # Single run settings
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--prim_lr', default=0.1, type=float, help='Learning rate for primary network')
    parser.add_argument('--sec_lr', default=None, type=float, help='Learning rate for adversary network')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--seed_run', action='store_true', help='Whether this is part of a run with multiple seeds')
    parser.add_argument('--seed_run_version', default=0, type=int, help='Version of the run with multiple seeds')
    
    # More general train settings
    parser.add_argument('--pretrain_steps', default=250, type=int)
    parser.add_argument('--opt', choices=OPT_BY_NAME.keys(), default="Adagrad", help='Name of optimizer')
    parser.add_argument('--log_dir', default='training_logs', type=str)
    parser.add_argument('--p_bar', action='store_true', help='Whether to use progressbar')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of crossvalidation folds')
    parser.add_argument('--no_grid_search', action='store_false', default=True, dest="grid_search", help='Don\'t optimize batch size and lr via gridsearch')
    parser.add_argument('--eval_batch_size', default=512, type=int, help='Batch size for evaluation. No effect on training or results, set as large as memory allows to maximize performance')
    parser.add_argument('--tf_mode', action='store_true', default=False, help='Use tensorflow rather than PyTorch defaults where possible. Only supports AdaGrad optimizer.')
    parser.add_argument('--no_early_stopping', action='store_true', default=False, help='Disable early stopping')
    parser.add_argument('--version', default=None, type=str, help='Override version. Default is the current time. Will be used in other scripts which call main.main().')

    # Dataset settings
    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS', 'EMNIST_35', 'EMNIST_10'], required=True)
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers that are used in dataloader')
    parser.add_argument('--disable_warnings', action='store_true', help='Whether to disable warnings about mean and std in the dataset')
    parser.add_argument('--sensitive_label', default=False, action='store_true', help='If True, target label will be included in list of sensitive columns; used for IPW(S+Y)')

    # ray settings
    parser.add_argument('--num_cpus', default=1, type=int, help='Number of CPUs used for each trial')
    parser.add_argument('--num_gpus', default=1, type=float, help='Number of GPUs used for each trial')

    args: argparse.Namespace = parser.parse_args()

    args.working_dir = os.getcwd()

    # run main loop
    main(args)
    
