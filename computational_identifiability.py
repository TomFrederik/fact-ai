import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from datasets import TabularDataset, CustomSubset

import numpy as np

import argparse
from time import time
import json


OPT_BY_NAME = {'Adagrad': torch.optim.Adagrad, 'Adam': torch.optim.Adam}


class Linear(pl.LightningModule):

    def __init__(self, num_features, lr, train_index2value, test_index2value, target_grp, optimizer):
        super().__init__()

        #self.save_hyperparameters()

        self.lr = lr
        self.train_index2value = train_index2value
        self.test_index2value = test_index2value
        self.target_grp = target_grp
        self.optimizer = optimizer
        #print(index2value)
        # {0: ('Other', 'Other'), 1: ('Other', 'Female'), 2: ('Black', 'Other'), 3: ('Black', 'Female')}

        self.net = nn.Linear(num_features, 1)
            
        # init loss
        self.loss_fct = nn.BCEWithLogitsLoss()



    def forward(self, x, y):

        input = torch.cat([x, y.unsqueeze(1)], dim=1).float()

        out = self.net(input).squeeze(dim=-1)

        return out
    
    def training_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s).float()

        loss = self.loss_fct(pred, targets) # CHECK THIS

        accuracy = torch.sum(torch.round(torch.sigmoid(pred)) == targets) / targets.shape[0]

        self.log('training/loss', loss)
        self.log('training/accuracy', accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s).float()

        loss = self.loss_fct(pred, targets) # CHECK THIS

        accuracy = torch.sum(torch.round(torch.sigmoid(pred)) == targets) / targets.shape[0]

        self.log('validation/loss', loss)
        self.log('validation/accuracy', accuracy)


        return loss

    def test_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s, test=True).float()

        loss = self.loss_fct(pred, targets) # CHECK THIS

        accuracy = torch.sum(torch.round(torch.sigmoid(pred)) == targets) / targets.shape[0]

        self.log('test/loss', loss)
        self.log('test/accuracy', accuracy)

        return loss
    
    def configure_optimizers(self):

        optimizer = self.optimizer(self.net.parameters(), lr=self.lr)

        return optimizer

    def idx_mapping(self, x, test=False):
        out = torch.zeros_like(x)

        if test:
            if self.target_grp == 'race':
                for key in self.test_index2value:
                    if self.test_index2value[key][0] == 'Black':
                        out[x == key] = 1
                return out
            elif self.target_grp == 'sex':
                for key in self.test_index2value:
                    if self.test_index2value[key][1] == 'Female':
                        out[x == key] = 1
                return out
            else:
                raise ValueError(f'Unexpected value for target_grp: {self.target_group}')
        else:
            if self.target_grp == 'race':
                for key in self.train_index2value:
                    if self.train_index2value[key][0] == 'Black':
                        out[x == key] = 1
                return out
            elif self.target_grp == 'sex':
                for key in self.train_index2value:
                    if self.train_index2value[key][1] == 'Female':
                        out[x == key] = 1
                return out
            else:
                raise ValueError(f'Unexpected value for target_grp: {self.target_group}')
        


def main(args):

    #TODO: do a gridsearch?

    # set run version
    args.version = str(int(time()))
    
    # seed RNG
    pl.seed_everything(args.seed)
    np.random.seed(args.seed)

    ## create train, val, test dataset
    # TODO: do this for images as well?
    dataset = TabularDataset(args.dataset, disable_warnings=args.disable_warnings, suffix=args.suffix)
    test_dataset = TabularDataset(args.dataset, test=True, disable_warnings=args.disable_warnings, suffix=args.suffix)
    
    # train val split
    all_idcs = np.random.permutation(np.arange(0, len(dataset), 1))
    train_idcs = all_idcs[:int(0.9 * len(all_idcs))]
    val_idcs = all_idcs[int(0.9 * len(all_idcs)):]
    train_dataset = CustomSubset(dataset, train_idcs)
    val_dataset = CustomSubset(dataset, val_idcs)
    
    # set up dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    # set up model
    model = Linear(num_features=dataset.dimensionality + 1, # + 1 for labels
                   lr=args.learning_rate,
                   train_index2value=dataset.protected_index2value,
                   test_index2value=test_dataset.protected_index2value,
                   target_grp=args.target_grp,
                   optimizer=OPT_BY_NAME[args.optimizer]) 

    if args.tf_mode:
        def init_weights(layer):
            if type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        model.apply(init_weights)
    
    # set up logger
    logdir=f'training_logs/{args.dataset}/identifiability/{args.target_grp}_version_{args.version}'
    logger = TensorBoardLogger(
        save_dir='./',
        name=logdir,
        version=''
    )

    # set up callbacks
    early_stopping = EarlyStopping(
        monitor='validation/accuracy',
        min_delta=0.0,
        patience=10,
        verbose=True,
        mode='max'
    )

    checkpoint = ModelCheckpoint(save_weights_only=True,
                                dirpath=logger.log_dir, 
                                mode='max', 
                                verbose=False,
                                monitor='validation/accuracy')

    callbacks = [early_stopping, checkpoint]

    # set up trainer
    trainer = pl.Trainer(logger=logger,
                        max_steps=args.train_steps,
                        callbacks=callbacks
                        )
    
    # train model
    fit_time = time()
    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    print(f'time to fit was {time()-fit_time}')

    # eval best model on test set
    trainer.test(test_dataloaders=test_loader, ckpt_path='best')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS'], required=True)
    parser.add_argument('--optimizer', choices=['Adagrad', 'Adam'], default='Adagrad')
    parser.add_argument('--target_grp', choices=['race', 'sex'], required=True, help='Whether to predict race or sex of a person')
    parser.add_argument('--suffix', default='', help='Dataset suffix to specify other datasets than the defaults')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--disable_warnings', action='store_true', help='Whether to disable warnings about mean and std in the dataset')
    parser.add_argument('--tf_mode', action='store_true', default=False, help='Use tensorflow rather than PyTorch defaults where possible. Only supports AdaGrad optimizer.')
    
    args = parser.parse_args()

    main(args)