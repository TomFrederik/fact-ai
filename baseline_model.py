from typing import Dict, Type, Optional, Any, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaselineModel(pl.LightningModule):

    def __init__(self,
        config: Dict[str, Any],
        num_features: int,
        hidden_units: List[int] = [64,32],
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
        opt_kwargs: Dict[str, Any] = {}
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
        net_list: List[torch.nn.Module] = []
        num_units = [self.hparams.num_features] + self.hparams.hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)

        # init loss
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.net(input).squeeze(dim=-1)
        return out
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        
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
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        self.log('validation/loss', loss)        
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        self.log('test/loss', loss)        

    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.config['lr'], **self.hparams.opt_kwargs)
