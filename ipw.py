from typing import Dict, Type, Optional, Any, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from arl import Learner



class IPW(pl.LightningModule):
    """Feed forward neural network with modified BCE loss, based on inverse 
    probability weighting of the losses.

    Attributes:
        config: Dict with hyperparameters (learning rate, batch size).
        num_features: Dimensionality of the data input.
        group_probs: Empirical observation probabilities of the different 
            protected groups.
        hidden_units: Number of hidden units in each layer of the network.
        optimizer: Optimizer used to update the model parameters.
        sensitive_label: Option to use joint probability of label and group
            membership for computing the weights.
        opt_kwargs: Optional; optimizer keywords other than learning rate.
    """

    def __init__(self, 
        config: Dict[str, Any],
        num_features: int,
        group_probs: torch.Tensor,
        hidden_units: List[int] = [64,32],
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
        sensitive_label: bool = False,
        opt_kwargs: Dict[str, Any] = {},
        ):
        """Inits an instance of IPW with the given attributes."""
        
        super().__init__()

        # save params EXCEPT group_probs since that throws an error
        self.save_hyperparameters('config', 'num_features', 'hidden_units', 'optimizer', 'sensitive_label', 'opt_kwargs')

        # save group probabilities
        self.group_probs = group_probs
        
        # init networks
        self.learner = Learner(input_shape=num_features, hidden_units=hidden_units)

        # init loss function
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Returns and logs the loss on the training set.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
            optimizer_idx: Index of the optimizer that is used for updating the 
                weights after the training step; 0 = learner, 1 = adversary.
        """
        
        x, y, s = batch

        loss = self.learner_step(x, y, s)

        # logging
        self.log("training/loss", loss)

        return loss

    
    def learner_step(self, x: torch.Tensor, y: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes the inversely reweighted or unweighted BCE loss.
    
        Args:
            x: Tensor of shape [batch_size, num_features] with data inputs.
            y: Tensor of shape [batch_size] with labels.
            s: Optional; tensor of shape [batch_size] with group indices.
    
        Returns:
            One of the following:
                
            The mean of single BCE losses that are reweighted with the inverse
            of the joint probabilities of labels and group memberships.
            
            The mean of single BCE losses that are reweighted with the inverse
            of the group probabilities.
            
            The unweighted BCE loss.
        """
        
        # compute unweighted bce
        logits = self.learner(x)
        bce = self.loss_fct(logits, y)

        # consider both s and y for selecting probability
        if s is not None:
            # compute weights
            if self.hparams.sensitive_label:
                sample_weights = torch.index_select(torch.index_select(self.group_probs, 0, s), 1, y.long())
            else:
                sample_weights = torch.index_select(self.group_probs, 0, s)

            # compute reweighted loss
            loss = torch.mean(bce / sample_weights)
        
        else:
            # compute unweighted loss
            loss = torch.mean(bce)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Computes and logs the validation loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        """
        
        x, y, _ = batch
        loss = self.learner_step(x, y)
        
        # logging
        self.log("validation/loss", loss)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Computes and logs the test loss.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        """
        
        x, y, _ = batch 
        loss = self.learner_step(x, y)
        
        # logging
        self.log("test/loss", loss)

    def configure_optimizers(self):
        """Chooses optimizer and learning-rate to use during optimization.
        
        Returns:
            Optimizer.       
        """
        
        optimizer = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.config['lr'], **self.hparams.opt_kwargs)

        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation of inputs through the network.
    
        Args:
            input: Tensor of shape [batch_size, num_features] with data inputs.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """
        return self.learner(x)
