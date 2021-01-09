import torch
import torch.nn as nn
import pytorch_lightning as pl



class ARL(pl.LightningModule):

    def __init__(self, 
        num_features,
        pretrain_steps,
        prim_hidden=[64,32],
        adv_hidden=[],
        prim_lr=0.01,
        adv_lr=0.01,
        optimizer=torch.optim.Adagrad,
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
        
        # init networks
        self.learner = Learner(num_features=num_features, hidden_units=prim_hidden)
        self.adversary = Adversary(num_features=num_features, hidden_units=adv_hidden)

        # init loss function
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    
    def training_step(self, batch, batch_idx, optimizer_idx):
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
        
        x, y, _ = batch         
        y = y.float()   # TODO: fix in datasets.py?

        if optimizer_idx == 0:
            loss = self.learner_step(x, y)
            
            # logging
            # TODO: add AUC metric
            self.log("training/reweighted_loss_learner", loss)
            
            return loss
            
        elif optimizer_idx == 1 and self.global_step > self.hparams.pretrain_steps:
            loss = self.adversary_step(x, y)
            
            return loss
    
    
    def learner_step(self, x, y):
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
        
        # compute lambdas
        lambdas = self.adversary(x)
        
        # compute reweighted loss  
        loss = torch.mean(lambdas * bce)
        
        return loss
     
        
    def adversary_step(self, x, y):
        '''        
        Inputs
        ----------
        x - float tensor of shape [batch_size, num_features], input features of the data batch
        y - int tensor of shape [batch_size], labels of the data batch

        Returns
        -------
        loss - scalar, minimization objective for the adversary        
        '''
        # compute unweighted bce
        logits = self.learner(x)        
        bce = self.loss_fct(logits, y)
        
        # compute lambdas
        lambdas = self.adversary(x)
        
        # compute reweighted loss
        loss = -torch.mean(lambdas * bce)
        
        return loss        
        
        
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch        
        y = y.float()   # TODO: fix in datasets.py?
        loss = self.learner_step(x, y)
        
        # logging
        # TODO: add AUC metric
        self.log("validation/reweighted_loss_learner", loss)

        
    def test_step(self, batch, batch_idx):
        x, y, _ = batch 
        y = y.float()   # TODO: fix in datasets.py?
        loss = self.learner_step(x, y)
        
        # logging
        # TODO: add AUC metric
        self.log("test/reweighted_loss_learner", loss)
 
    
    def configure_optimizers(self):
        '''
        Returns
        -------
        [optimizer_learn, optimizer_adv] - list, optimizers for learner and adversary
        [] - list, learning rate schedulers for learner and adversary (not used)
        '''
        
        # Create optimizers for learner and adversary
        optimizer_learn = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.prim_lr, **self.hparams.opt_kwargs)
        optimizer_adv = self.hparams.optimizer(self.adversary.parameters(), lr=self.hparams.adv_lr, **self.hparams.opt_kwargs)

        return [optimizer_learn, optimizer_adv], []

    def forward(self, x):
        return self.learner(x)
    
    
class Learner(nn.Module):
    def __init__(self, 
        num_features,
        hidden_units=[64,32]
        ):        
        '''
        num_features - int, number of features of the input
        hidden_units - list, number of hidden units in each layer of the MLP
        '''
            
        super().__init__()
        
        # construct network
        net_list = []
        num_units = [num_features] + hidden_units
        for i in range(len(num_units)-1):
            net_list.append(nn.Linear(num_units[i],num_units[i+1]))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)
        
    def forward(self, x):
        '''        
        Inputs
        ----------
        x - float tensor of shape [batch_size, num_features], input features of the data batch

        Returns
        -------
        out - float tensor of shape [batch_size], logits of the data batch under the learner network        
        '''
        
        out = self.net(x)
        
        return torch.squeeze(out, dim=-1)

    
class Adversary(nn.Module):
    def __init__(self, 
        num_features,
        hidden_units=[]
        ):
        
        '''
        num_features - int, number of features of the input
        hidden_units - list, number of hidden units in each layer of the MLP
        '''
            
        super().__init__()
        
        # construct network
        net_list = []
        num_units = [num_features] + hidden_units
        for i in range(len(num_units)-1):   # range = 0 for one linear layer (hidden_units list is empty)
            net_list.append(nn.Linear(num_units[i],num_units[i+1]))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))
        net_list.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_list)
        
    def forward(self, x):
        '''        
        Inputs
        ----------
        x - float tensor of shape [batch_size, num_features], input features of the data batch

        Returns
        -------
        out - float tensor of shape [batch_size], lambdas for reweighting the loss        
        '''
        
        # compute adversary
        adv = self.net(x)
        
        # normalize adversary across batch
        # TODO: check numerical stability
        adv_norm = adv / torch.sum(adv)
        
        # scale and shift
        out = x.shape[0] * adv_norm + torch.ones_like(adv_norm)        
        
        return torch.squeeze(out)
    
    
    
