import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import baseline_model
from torch.utils.data import DataLoader

from datasets import Dataset
from arl import ARL
from baseline_model import BaselineModel
from auc import AUCLogger

import argparse
import os

# dict to access optimizers by name, if we need to use different opts.
OPT_BY_NAME = {'Adagrad': torch.optim.Adagrad}

#TODO could be replaced by calculating the metrics in the validation steps of the models
class MetricCallback(pl.Callback):
    """
    The MetricCallback class calculates relevant metrics
    and saves them to the PyTorch Lightning training session
    """

    def __init__(self, test_loader):
        """
        Initializes the metric callback instance
        :param test_loader: dataloader used for evaluation on test set
        """

        super().__init__()
        self.test_loader = test_loader

    def on_epoch_end(self, trainer, pl_module):
        """
        Called after every epoch
        :param trainer: the pl trainer instance
        :param pl_module: the model
        :return: not implemented
        """

        self.evaluate_metric(trainer, pl_module, trainer.current_epoch+1)

    def evaluate_metric(self, trainer, pl_module, step):
        """
        #TODO
        :param trainer: the pl trainer instance
        :param pl_module: the model
        :param step: the current train step number
        :return: not implemented
        """

        # TODO calculate metric here or call corresponding function
        trainer.logger.experiment.add_scalar(placeholder_name, placeholder_var, step)


def train(args):
    """
    Function to train a model
    :param args: object from the argument parser
    :return: Results from testing the model
    """

    # Seed for reproducability
    pl.seed_everything(args.seed)

    # Create datasets
    train_dataset = Dataset(args.dataset)
    test_dataset = Dataset(args.dataset, test=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    # Select model and instantiate
    if args.model == 'ARL':
        model = ARL(num_features=train_dataset.dimensionality,
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


    os.makedirs(args.log_dir, exist_ok=True)


    # Create a PyTorch Lightning trainer
    metric_callback = MetricCallback(test_loader)
    auc_callback = AUCLogger(test_dataset)
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_steps=args.train_steps,
                         #callbacks=[metric_callback],  #TODO enable?
                         callbacks=[auc_callback],
                         progress_bar_refresh_rate=1 if args.p_bar else 0)

    # Training
    #metric_callback.evaluate_metric(trainer, model, step=0)  #TODO enable?
    trainer.fit(model, train_loader, test_loader)

    # Testing
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)

    return test_result


if __name__ == "__main__":
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO'], required=True)
    parser.add_argument('--prim_hidden', default=[64, 32], help='Number of hidden units in primary network')
    parser.add_argument('--adv_hidden', default=[32], help='Number of hidden units in adversarial network')

    # Training settings
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--test_batch_size', default=100, type=int)
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
