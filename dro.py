# Adapted from https://worksheets.codalab.org/worksheets/0x17a501d37bbe49279b0c70ae10813f4c/

from typing import Dict, Type, Any, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl


class DRO_loss(nn.Module):
    """Provides the DRO loss under the given hyperparameters eta and k.

    Attributes:
        eta: Threshold for single losses that contribute to learning objective.
        k: Exponent to upweight high losses.
    """
    
    def __init__(self, eta: float, k: float):
        """Inits an instance of DRO_loss with the given hyperparameters."""
        super(DRO_loss, self).__init__()
        self.eta = eta
        self.k = k
        self.logsig = nn.LogSigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes DRO loss from logits and labels.
    
        Args:
            x: Tensor of shape [batch_size] with logits under the given model.
            y: Tensor of shape [batch_size] with ground truth (0 or 1). 
    
        Returns:
            The DRO loss (modified Binary Cross-Entropy loss) if k is positive,
            otherwise the Binary Cross-Entropy loss.
        """
        
        bce = -1*y*self.logsig(x) - (1-y)*self.logsig(-x)

        if self.k > 0:
            bce = self.relu(bce - self.eta)            
            bce = bce**self.k
            return bce.mean()
        else:
            return bce.mean()



class DRO(pl.LightningModule):
    """Feed forward neural network with a modified BCE loss based on 
    distributionally robust optimization.

    Attributes:
        config: Dict with hyperparameters learning rate, batch size, eta.
        num_features: Dimensionality of the data input.
        pretrain_steps: Number of pretraining steps before using the DRO loss.
        hidden_units: Number of hidden units in each layer of the network.
        k: Exponent to use for computing the DRO loss.
        optimizer: Optimizer used to update the model parameters.
        opt_kwargs: Optional; optimizer keywords other than learning rate.
    """

    def __init__(self, 
        config: Dict[str, Any],
        num_features: int,
        pretrain_steps: int,
        hidden_units: List[int] = [64,32],
        k=2.0,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
        opt_kwargs: Dict[str, Any] = {},
        ):
        """Inits an instance of DRO with the given attributes."""
        
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

        # init DRO loss
        self.loss_fct = DRO_loss(self.hparams.config['eta'], self.hparams.k)
        
        # init pretrain loss
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward propagation of inputs through the network.
    
        Args:
            input: Tensor of shape [batch_size, num_features] with data inputs
    
        Returns:
            Tensor of shape [batch_size] with predicted logits
        """
        
        out = self.net(input).squeeze(dim=-1)
        return out
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch
            batch_idx: Index of batch in the dataset (not needed)
    
        Returns:
            BCE loss of the batch on the training dataset during pretraining, 
            DRO loss after pretraining. 
        """
        
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)
        
        # compute loss
        if self.global_step > self.hparams.pretrain_steps:        
            loss = self.loss_fct(logits, y)
        else:
            loss = self.bce(logits, y)

        # logging
        self.log('training/loss', loss)

        return loss        
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Compute and log the validation loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch
            batch_idx: Index of batch in the dataset (not needed)
        """
        
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        if self.global_step > self.hparams.pretrain_steps:        
            loss = self.loss_fct(logits, y)
        else:
            loss = self.bce(logits, y)

        # logging
        self.log('validation/loss', loss)        
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Compute and log the test loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch
            batch_idx: Index of batch in the dataset (not needed)
        """
        
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        if self.global_step > self.hparams.pretrain_steps:        
            loss = self.loss_fct(logits, y)
        else:
            loss = self.bce(logits, y)

        # logging
        self.log('test/loss', loss)        

    
    def configure_optimizers(self):
        """Choose optimizer and learning-rate to use during optimization.
        
        Return:
            Optimizer           
        """
        
        return self.optimizer(self.parameters(), lr=self.hparams.config['lr'], **self.hparams.opt_kwargs)
