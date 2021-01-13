import torch
import torch.nn as nn
import pytorch_lightning as pl
from arl import Learner



class IPW(pl.LightningModule):

    def __init__(self, 
        config,
        num_features,
        hidden_units=[64,32],
        lr=0.01, # deprecated
        optimizer=torch.optim.Adagrad,
        group_probs=None,
        sensitive_label=False,
        opt_kwargs={},
        ):
        """
        Class for inverse probability weighting
        :param num_features: int, number of features of the input
        :param hidden_units: list, number of hidden units in each layer of the learner network
        :param lr: float, learning rate for updating the learner
        :param optimizer: torch.optim.Optimizer constructor function, optimizer to adjust the model's parameters
        :param group_probs: empirical observation probabilities of the different protected groups
        :param opt_kwargs: dict, optimizer keywords (other than learning rate)
        """
        super().__init__()

        # save params
        self.save_hyperparameters()

        # save group probabilities
        self.group_probs = group_probs
        
        # init networks
        self.learner = Learner(num_features=num_features, hidden_units=hidden_units)

        # init loss function
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    
    def training_step(self, batch, batch_idx):
        """
        Implements the training step for PyTorch Lightning
        :param batch: input batch from dataset
        :param batch_idx: index of batch in the dataset (not needed)
        :return: scalar, minimization objective
        """
        x, y, s = batch
        y = y.float()   # TODO: fix in datasets.py?

        loss = self.learner_step(x, y, s)

        # logging
        self.log("training/loss", loss)

        return loss

    
    def learner_step(self, x, y, s=None):
        """
        TODO
        :param x: TODO
        :param y: TODO
        :param s: TODO
        :return: TODO
        """
        # compute unweighted bce
        logits = self.learner(x)
        bce = self.loss_fct(logits, y)

        # consider both s and y for selecting probability

        if s is not None:
            # compute weights
            if self.hparams.sensitive_label:
                sample_weights = torch.index_select(torch.index_select(self.group_probs, 0, s), 0, y)
            else:
                sample_weights = torch.index_select(self.group_probs, 0, s)

            # compute reweighted loss
            loss = torch.mean(bce / sample_weights)
        else:
            # compute unweighted loss
            loss = torch.mean(bce)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        TODO
        :param batch: TODO
        :param batch_idx: TODO
        :return: TODO
        """
        x, y, _ = batch
        y = y.float()   # TODO: fix in datasets.py?
        loss = self.learner_step(x, y)
        
        # logging
        self.log("validation/loss", loss)

    def test_step(self, batch, batch_idx):
        """
        TODO
        :param batch: TODO
        :param batch_idx: TODO
        :return: TODO
        """
        x, y, _ = batch 
        y = y.float()   # TODO: fix in datasets.py?
        loss = self.learner_step(x, y)
        
        # logging
        self.log("test/loss", loss)

    def configure_optimizers(self):
        """
        TODO
        :return: TODO
        """
        # Create optimizers for learner and adversary
        optimizer = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.config['lr'], **self.hparams.opt_kwargs)

        return optimizer

    def forward(self, x):
        """
        TODO
        :param x: TODO
        :return: TODO
        """
        return self.learner(x)
