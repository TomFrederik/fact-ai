import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaselineModel(pl.LightningModule):

    def __init__(self, 
    num_features,
    hidden_units=[64,32],
    lr=0.01,
    optimizer=torch.optim.Adagrad,
    opt_kwargs={},
    ):
        '''
        num_features - int, number of features of the input
        hidden_units - list, number of hidden units in each layer of the DNN
        lr - float, learning rate
        optimizer - torch.optim.Optimizer constructor function, optimizer to adjust the model's parameters
        opt_kwargs - dict, optimizer keywords (other than learning rate)
        '''
        
        super().__init__()

        # save params
        self.save_hyperparameters()

        self.optimizer = optimizer

        # construct network
        net_list = []
        num_units = [self.hparams.num_features] + self.hparams.hidden_units
        for i in range(len(num_units)-1):
            net_list.append(nn.Linear(num_units[i],num_units[i+1]))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)

        # init loss
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input):
        out = self.net(input).squeeze_()
        return out
    
    def training_step(self, batch, batch_idx):
        
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        #TODO: add other metrics
        self.log('training/loss', loss)

        return loss        
        
    def validation_step(self, batch, batch_idx):
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        #TODO: add other metrics
        self.log('validation/loss', loss)        
        
    def test_step(self, batch, batch_idx):
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        #TODO: add other metrics
        self.log('test/loss', loss)        

    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.lr, **self.hparams.opt_kwargs)
