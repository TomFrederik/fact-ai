import torch
import torch.nn as nn
import pytorch_lightning as pl



class ARL(pl.LightningModule):

    def __init__(self, 
    num_features,
    prim_hidden=[64,32],
    adv_hidden=[32],
    prim_lr=0.01,
    adv_lr=0.01,
    batch_size=256,
    optimizer=torch.optim.Adagrad,
    opt_kwargs={},
    ):
        '''
        num_features - int, number of features of the input
        prim_hidden - list, number of hidden units in each layer of the learner network
        adv_hidden - list, number of hidden units in each layer of the adversary network
        prim_lr - float, learning rate for updating the learner
        adv_lr - float, learning rate for updating the adversary
        batch_size - int, batch size that is used to iterate through the dataset
        optimizer - torch.optim.Optimizer constructor function, optimizer to adjust the model's parameters
        opt_kwargs - dict, optimizer keywords (other than learning rate)
        '''
        
        super().__init__()

        # save params
        self.save_hyperparameters()
        
        # init networks
        self.learner = Learner(num_features=num_features, hidden_dims=prim_hidden)
        self.adversary = Adversary(num_features=num_features, hidden_dims=adv_hidden)
        
        # init lambdas
        self.lambdas = torch.ones(self.hparams.batch_size, device=self.device)
        
        # init unweighted binary cross entropy loss
        # TODO: change initialization? shouldn't matter as long as learner is called first
        self.bce = torch.zeros(self.hparams_batch_size, device=self.device)

        # init loss function
        self.loss_fct = nn.BCEWithLogitsLoss()

    
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
        loss - scalar, minimization objective,
        '''
        x, y = batch

        if optimizer_idx == 0:
            loss = self.learner_step(x, y)
        elif optimizer_idx == 1:
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
        self.bce = self.loss_fct(logits, y, reduce=None)
        
        # compute reweighted loss        
        loss = torch.dot(self.lambdas, self.bce)
        
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
        # compute lambdas
        self.lambdas = self.adversary(x)
        
        # compute reweighted loss
        loss = -torch.dot(self.lambdas, self.bce)
        
        return loss        
        
        
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError 

        
    def test_step(self, batch, batch_idx):
        raise NotImplementedError
 
    
    def configure_optimizers(self):
        # Create optimizers for learner and adversary
        optimizer_learn = self.optimizer(self.learner.parameters(), lr=self.hparams.prim_lr)
        optimizer_adv = self.optimizer(self.adversary.parameters(), lr=self.hparams.adv_lr)

        return [optimizer_learn, optimizer_adv]
    

    
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
        return out


    
class Adversary():
    raise NotImplementedError