import torch
import torch.nn as nn
import pytorch_lightning as pl
from arl import Learner



class IPW(pl.LightningModule):

    def __init__(self, 
        num_features,
        hidden_units=[64,32],
        lr=0.01,
        optimizer=torch.optim.Adagrad,
        group_probs=None,
        opt_kwargs={},
        ):
        '''
        num_features - int, number of features of the input
        prim_hidden - list, number of hidden units in each layer of the learner network
        adv_hidden - list, number of hidden units in each layer of the adversary network
        prim_lr - float, learning rate for updating the learner
        adv_lr - float, learning rate for updating the adversary
        optimizer - torch.optim.Optimizer constructor function, optimizer to adjust the model's parameters
        opt_kwargs - dict, optimizer keywords (other than learning rate)
        '''
        
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
        '''        
        Inputs
        ----------
        batch - input batch from dataset 
        batch_idx - index of batch in the dataset (not needed)
        optimizer_idx - index of the optimizer to use for the training step,
                        0 = learner, 1 = adversary
            
        Returns
        -------
        loss - scalar, minimization objective
        '''
        
        x, y, s = batch
        y = y.float()   # TODO: fix in datasets.py?

        loss = self.learner_step(x, y, s)

        # logging
        self.log("training/IPW", loss)

        return loss

    
    def learner_step(self, x, y, s=None):
        '''        
        Inputs
        ----------
        x - float tensor of shape [batch_size, num_features], input features of data batch
        y - int tensor of shape [batch_size], labels of data batch

        Returns
        -------
        loss - scalar, minimization objective for the learner       
        '''
        
        # compute unweighted bce
        logits = self.learner(x)
        bce = self.loss_fct(logits, y)

        if s is not None:
            # compute weights
            sample_weights = torch.index_select(self.group_probs, 0, s)
            # compute reweighted loss
            loss = torch.mean(bce / sample_weights)
        else:
            # compute unweighted loss
            loss = torch.mean(bce)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.float()   # TODO: fix in datasets.py?
        loss = self.learner_step(x, y)
        
        # logging
        self.log("validation/IPW", loss)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch 
        y = y.float()   # TODO: fix in datasets.py?
        loss = self.learner_step(x, y)
        
        # logging
        self.log("test/IPW", loss)

    def configure_optimizers(self):
        '''
        Returns
        -------
        [optimizer_learn, optimizer_adv] - list, optimizers for learner and adversary
        [] - list, learning rate schedulers for learner and adversary (not used)
        '''
        
        # Create optimizers for learner and adversary
        optimizer = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.lr, **self.hparams.opt_kwargs)

        return optimizer

    def forward(self, x):
        return self.learner(x)
