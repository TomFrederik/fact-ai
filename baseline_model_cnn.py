from typing import Dict, Type, Optional, Any, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaselineModel(pl.LightningModule):
    """Fully-connected feed forward neural network.

    Attributes:
        config: Dict with hyperparameters (learning rate, batch size).
        num_features: Dimensionality of the data input.
        hidden_units: Number of hidden units in each layer of the network.
        optimizer: Optimizer used to update the model parameters.
        opt_kwargs: Optional; optimizer keywords other than learning rate.
    """

    def __init__(self,
        config: Dict[str, Any],
        num_features: int,
        hidden_units: List[int] = [64,32],
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
        opt_kwargs: Dict[str, Any] = {}
        ):
        """Inits an instance of the network with the given attributes."""
        
        super().__init__()

        # save params
        self.save_hyperparameters()

        self.optimizer = optimizer

        # construct network
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
                                nn.MaxPool2d(kernel_size=(2, 2)),
                                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                                nn.MaxPool2d(kernel_size=(2, 2)),
                                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
                                nn.MaxPool2d(kernel_size=(2, 2)),
                                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3)),
                                nn.MaxPool2d(kernel_size=(2, 2)),
                                nn.Flatten())
        net_list: List[torch.nn.Module] = []
        num_units = [2304] + self.hparams.hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)

        # init loss
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward propagation of inputs through the network.
    
        Args:
            input: Tensor of shape [batch_size, num_features] with data inputs.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """
        intermediate = self.cnn(input)
        out = self.net(intermediate).squeeze(dim=-1)
        return out
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Computes and logs the training loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
    
        Returns:
            BCE loss of the batch on the training dataset. 
        """
        
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        self.log('training/loss', loss)

        return loss        
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Computes and logs the validation loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        """
        
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        self.log('validation/loss', loss)        
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Computes and logs the test loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        """
        
        # get features and labels
        x, y, s = batch
        
        # compute logits
        logits = self(x)

        # compute loss
        loss = self.loss_fct(logits, y)

        # logging
        self.log('test/loss', loss)        

    
    def configure_optimizers(self):
        """Chooses optimizer and learning-rate to use during optimization.
        
        Returns:
            Optimizer.       
        """
        
        return self.optimizer(self.parameters(), lr=self.hparams.config['lr'], **self.hparams.opt_kwargs)
