import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from datasets import TabularDataset, CustomSubset, EMNISTDataset

import numpy as np

import argparse
from time import time
import json


OPT_BY_NAME = {'Adagrad': torch.optim.Adagrad, 'Adam': torch.optim.Adam}


class Linear(pl.LightningModule):

    def __init__(self, num_features, lr, train_index2value, test_index2value, target_grp, optimizer, dataset_type, strength):
        super().__init__()

        #self.save_hyperparameters()

        self.lr = lr
        self.train_index2value = train_index2value
        self.test_index2value = test_index2value
        self.target_grp = target_grp
        self.optimizer = optimizer
        self.dataset_type = dataset_type
        #print(index2value)
        # {0: ('Other', 'Other'), 1: ('Other', 'Female'), 2: ('Black', 'Other'), 3: ('Black', 'Female')}

        if self.dataset_type == 'tabular':
            self.net = nn.Linear(num_features, 1)
        elif self.dataset_type == 'image':
            # construct network
            if strength == 'weak':
                self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3)),
                                         nn.MaxPool2d(kernel_size=(2, 2)),
                                         nn.Flatten())
                self.fc = nn.Linear(338 + 1, 1)
            elif strength == 'normal':
                self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
                                         nn.MaxPool2d(kernel_size=(2, 2)),
                                         nn.Flatten())
                self.fc = nn.Linear(5408 + 1, 1)
            elif strength == 'strong':
                self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3)),
                                         nn.MaxPool2d(kernel_size=(2, 2)),
                                         nn.Flatten())
                self.fc = nn.Linear(10816 + 1, 1)
            elif strength == 'stronger':
                self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3)),
                                         nn.MaxPool2d(kernel_size=(2, 2)),
                                         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
                                         nn.MaxPool2d(kernel_size=(2, 2)),
                                         nn.Flatten())
                self.fc = nn.Linear(3200 + 1, 1)
            else:
                raise Exception("Strength of the Adversary CNN not recognized!")
        else:
            raise Exception(f"Model was unable to recognize dataset type {self.dataset_type}!")
            
        # init loss
        self.loss_fct = nn.BCEWithLogitsLoss()



    def forward(self, x, y):

        if self.dataset_type == 'tabular':
            input = torch.cat([x, y.unsqueeze(1)], dim=1).float()

            out = self.net(input).squeeze(dim=-1)

        else:
            intermediate = self.cnn(x)
            intermediate = torch.cat([intermediate.float(), y.float().unsqueeze(1)], dim=1)
            out = self.fc(intermediate).squeeze(dim=-1)

        return out
    
    def training_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s).float()

        loss = self.loss_fct(pred, targets) # CHECK THIS

        accuracy = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred)) == targets), targets.shape[0])

        self.log('training/loss', loss)
        self.log('training/accuracy', accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s).float()

        loss = self.loss_fct(pred, targets) # CHECK THIS

        accuracy = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred)) == targets), targets.shape[0])

        self.log('validation/loss', loss)
        self.log('validation/accuracy', accuracy)


        return loss

    def test_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s, test=True).float()

        loss = self.loss_fct(pred, targets)

        grp_1_idcs = targets == 0
        grp_2_idcs = targets == 1

        accuracy = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred)) == targets), targets.shape[0])
        accuracy_grp_1 = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred[grp_1_idcs])) == targets[grp_1_idcs]), targets[grp_1_idcs].shape[0])
        accuracy_grp_2 = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred[grp_2_idcs])) == targets[grp_2_idcs]), targets[grp_2_idcs].shape[0])
        
        self.log('test/loss', loss)
        self.log('test/accuracy', accuracy)
        self.log('test/accuracy_grp_1', accuracy_grp_1)
        self.log('test/accuracy_grp_2', accuracy_grp_2)

        return loss
    
    def configure_optimizers(self):

        optimizer = self.optimizer(self.parameters(), lr=self.lr)

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
            elif self.target_grp == 'dummy':
                for key in self.test_index2value:
                    if self.test_index2value[key] == 'protected':
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
            elif self.target_grp == 'dummy':
                for key in self.train_index2value:
                    if self.train_index2value[key] == 'protected':
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
    if args.dataset == 'EMNIST_35':
        assert args.target_grp == 'dummy', "Target group not recognized for EMNIST_35 dataset!"
        dataset = EMNISTDataset()
        test_dataset = EMNISTDataset(test=True)
    elif args.dataset == 'EMNIST_10':
        assert args.target_grp == 'dummy', "Target group not recognized for EMNIST_10 dataset!"
        dataset = EMNISTDataset(imb=True)
        test_dataset = EMNISTDataset(imb=True, test=True)
    else:
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
    model = Linear(num_features=dataset.dimensionality + 1 if args.dataset_type != 'image' else None, # + 1 for labels
                   lr=args.learning_rate,
                   train_index2value=dataset.protected_index2value,
                   test_index2value=test_dataset.protected_index2value,
                   target_grp=args.target_grp,
                   optimizer=OPT_BY_NAME[args.optimizer],
                   dataset_type = args.dataset_type,
                   strength=args.adv_cnn_strength)

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
                         callbacks=callbacks,
                         progress_bar_refresh_rate=1 if args.p_bar else 0,
                         )
    
    # train model
    fit_time = time()
    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    print(f'time to fit was {time()-fit_time}')

    # eval best model on test set
    return trainer.test(test_dataloaders=test_loader, ckpt_path='best')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS', 'EMNIST_35', 'EMNIST_10'], required=True)
    parser.add_argument('--adv_cnn_strength', choices=['weak', 'normal', 'stronger'], default='normal', help='One of the pre-set strength settings of the CNN Adversarial in ARL')
    parser.add_argument('--optimizer', choices=['Adagrad', 'Adam'], default='Adagrad')
    parser.add_argument('--target_grp', choices=['race', 'sex', 'dummy'], required=True, help='Whether to predict race or sex of a person')
    parser.add_argument('--suffix', default='', help='Dataset suffix to specify other datasets than the defaults')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--disable_warnings', action='store_true', help='Whether to disable warnings about mean and std in the dataset')
    parser.add_argument('--tf_mode', action='store_true', default=False, help='Use tensorflow rather than PyTorch defaults where possible. Only supports AdaGrad optimizer.')
    parser.add_argument('--p_bar', action='store_true', help='Whether to use progressbar')
    
    args = parser.parse_args()

    args.dataset_type = 'image' if args.dataset in ['EMNIST_35', 'EMNIST_10'] else 'tabular'

    main(args)