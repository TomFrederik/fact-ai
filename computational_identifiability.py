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
    """Linear model (or simple CNN) to predict group membership.

    Attributes:
        num_features: Dimensionality of the data input.
        lr: Learning rate.
        train_index2value: Dictionary mapping group index 0-3 to a tuple in race x sex used in the train set.
        test_index2value: Dictionary mapping group index 0-3 to a tuple in race x sex used in the test set.
        target_grp: A value from ['race', 'sex'], describing the variable to predict.
        optimizer: Optimizer used to update the model parameters.
        dataset_type: Indicator for which datatype is used.
        strength: Strength / Model capacity of the CNN.

    Raises:
        Exception: If the dataset type is neither tabular nor image data.
        Exception: If the strength of the CNN adversary was not recognized.
    """

    def __init__(self, num_features, lr, train_index2value, test_index2value, target_grp, optimizer, dataset_type, strength):
        """
        Instantiates the model with the given attributes.
        """
        super().__init__()

        #self.save_hyperparameters()

        self.lr = lr
        self.train_index2value = train_index2value
        self.test_index2value = test_index2value
        self.target_grp = target_grp
        self.optimizer = optimizer
        self.dataset_type = dataset_type

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
            else:
                raise Exception("Strength of the Adversary CNN not recognized!")
        else:
            raise Exception(f"Model was unable to recognize dataset type {self.dataset_type}!")
            
        # init loss
        self.loss_fct = nn.BCEWithLogitsLoss()



    def forward(self, x, y):
        """Forward propagation of inputs through the network.
    
        Args:
            x: Tensor of shape [batch_size, num_features] with data inputs.
            y: Tensor of shape [batch_size] with data labels.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits for group membership.
        """
        if self.dataset_type == 'tabular':
            input = torch.cat([x, y.unsqueeze(1)], dim=1).float()

            out = self.net(input).squeeze(dim=-1)

        else:
            intermediate = self.cnn(x)
            intermediate = torch.cat([intermediate.float(), y.float().unsqueeze(1)], dim=1)
            out = self.fc(intermediate).squeeze(dim=-1)

        return out
    
    def training_step(self, batch, batch_idx):
        """Computes and logs the training loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
    
        Returns:
            BCE loss of the batch on the training dataset. 
        """

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s).float()

        loss = self.loss_fct(pred, targets)

        accuracy = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred)) == targets), targets.shape[0])

        self.log('training/loss', loss)
        self.log('training/accuracy', accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Computes and logs the validation loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        
        Returns:
            BCE loss of the batch on the validation dataset. 
        """

        x, y, s = batch

        pred = self.forward(x, y)

        targets = self.idx_mapping(s).float()

        loss = self.loss_fct(pred, targets)

        accuracy = torch.true_divide(torch.sum(torch.round(torch.sigmoid(pred)) == targets), targets.shape[0])

        self.log('validation/loss', loss)
        self.log('validation/accuracy', accuracy)


        return loss

    def test_step(self, batch, batch_idx):
        """Computes and logs the test loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        
        Returns:
            BCE loss of the batch on the test dataset. 
        """

        x, y, s = batch

        pred = torch.round(torch.sigmoid(self.forward(x, y)))

        targets = self.idx_mapping(s, test=True).float()

        # compute scores
        loss = self.loss_fct(pred, targets)
        accuracy = torch.true_divide(torch.sum(pred == targets), targets.shape[0])
        
        self.log('test/loss', loss)
        self.log('test/accuracy', accuracy)

        return targets, pred
    
    def test_epoch_end(self, outputs):
        """ 
        Computest group specific accuracy scores.

        Args:
            outputs: List with elements targets and preds containing batch outputs of test_step
        """
        # extract test results
        targets = []
        preds = []
        
        for idx in range(len(outputs)):
            targets.append(outputs[idx][0])
            preds.append(outputs[idx][1])
        
        targets = torch.cat(targets, dim=0)
        preds = torch.cat(preds, dim=0)

        # compute group specific scores
        grp_1_idcs = targets == 0
        grp_2_idcs = targets == 1

        grp_1_targets = targets[grp_1_idcs]
        grp_2_targets = targets[grp_2_idcs]
        
        accuracy_grp_1 = torch.true_divide(torch.sum(preds[grp_1_idcs]== grp_1_targets), grp_1_targets.shape[0])
        accuracy_grp_2 = torch.true_divide(torch.sum(preds[grp_2_idcs] == grp_2_targets), grp_2_targets.shape[0])

        self.log('test/accuracy_grp_1', accuracy_grp_1)
        self.log('test/accuracy_grp_2', accuracy_grp_2)



    def configure_optimizers(self):
        """Chooses optimizer and learning-rate to use during optimization.
        
        Returns:
            Optimizer.       
        """

        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        return optimizer


    def idx_mapping(self, x, test=False):
        """
        Maps the group membership to a binary value, depending on the target group.

        Args:
            x: Tensor of shape [batch_size] with values between 0 and 3.
            test: Bool, whether to use the test index2value mapping.
        
        Returns:
            out: Tensor of shape [batch_size] containing the binarized group membership.
        """

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

    # set run version
    args.version = str(int(time()))
    
    # seed RNG
    pl.seed_everything(args.seed)
    np.random.seed(args.seed)

    # create train, val, test dataset
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
                   optimizer=OPT_BY_NAME[args.opt],
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
    parser.add_argument('--adv_cnn_strength', choices=['weak', 'normal', 'strong'], default='normal', help='One of the pre-set strength settings of the CNN Adversarial in ARL')
    parser.add_argument('--opt', choices=['Adagrad', 'Adam'], default='Adagrad')
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
