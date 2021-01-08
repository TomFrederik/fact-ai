import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.functional.classification import auroc
from torch.utils.data import DataLoader

from datasets import Dataset
from arl import ARL
from baseline_model import BaselineModel
from metrics import Logger

import argparse
import os

import numpy as np

from sklearn.model_selection import KFold

# dict to access optimizers by name, if we need to use different opts.
OPT_BY_NAME = {'Adagrad': torch.optim.Adagrad}


def train(args):
    """
    Function to train a model
    :param args: object from the argument parser
    :return: Results from testing the model
    """

    # create logdir
    logdir = os.path.join(args.log_dir, args.dataset, args.model)
    os.makedirs(logdir, exist_ok=True)

    # Seed for reproducability
    pl.seed_everything(args.seed)

    # Create datasets
    train_dataset = Dataset(args.dataset)
    test_dataset = Dataset(args.dataset, test=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    

    

    test_callback = Logger(test_dataset, 'test')
    
    # perform 5-fold crossvalidation
    kf = KFold(n_splits=5)
    fold_nbr = 0
    aucs = []
    for train, val in kf.split(train_dataset):
        fold_nbr += 1

        train_callback = Logger(train, 'training')
        val_callback = Logger(val, 'validation')

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


        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(default_root_dir=os.path.join(logdir,f'fold_{fold_nbr}'),
                            checkpoint_callback=ModelCheckpoint(save_weights_only=True),
                            gpus=1 if torch.cuda.is_available() else 0,
                            max_steps=args.train_steps+args.pretrain_steps,
                            callbacks=[train_callback, val_callback], # test callback?
                            progress_bar_refresh_rate=1 if args.p_bar else 0
                            #fast_dev_run=True # FOR DEBUGGING, SET TO FALSE FOR REAL TRAINING
                            )

        # Training
        trainer.fit(model, train_loader)

        # Evaluate on val set to get an estimate of performance
        scores = torch.sigmoid(model(val))
        aucs.append(auroc(scores, val.labels).item())
    
    mean_auc = np.mean(aucs)
        

    # Testing
    #model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    #test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)

    #return test_result
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
    parser.add_argument('--p_bar', action='store_true')

    # Dataset settings
    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS'], required=True)
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers that are used in dataloader')

    args = parser.parse_args()

    # run training loop
    train(args)
