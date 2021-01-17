from typing import Dict, Type, Optional, Any, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl



class ARL(pl.LightningModule):

    def __init__(self, 
        config: Dict[str, Any],
        num_features: int,
        pretrain_steps: int,
        prim_hidden: List[int] = [64,32],
        adv_hidden: List[int] = [],
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
        opt_kwargs: Dict[str, Any] = {},
        ):
        '''
        num_features - int, number of features of the input
        prim_hidden - list, number of hidden units in each layer of the learner network
        adv_hidden - list, number of hidden units in each layer of the adversary network
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

    
    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int,
                      optimizer_idx: int) -> Optional[torch.Tensor]:
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

        if optimizer_idx == 0:
            loss = self.learner_step(x, y)
            
            # logging
            self.log("training/reweighted_loss_learner", loss)
            
            return loss

        elif optimizer_idx == 1 and self.global_step > self.hparams.pretrain_steps:
            loss = self.adversary_step(x, y)
            
            return loss

        else:
            return None


    def learner_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
     
        
    def adversary_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        x, y, _ = batch        
        loss = self.learner_step(x, y)
        
        # logging
        self.log("validation/reweighted_loss_learner", loss)

        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        x, y, _ = batch 
        loss = self.learner_step(x, y)
        
        # logging
        self.log("test/reweighted_loss_learner", loss)
 
    
    def configure_optimizers(self):
        '''
        Returns
        -------
        [optimizer_learn, optimizer_adv] - list, optimizers for learner and adversary
        [] - list, learning rate schedulers for learner and adversary (not used)
        '''
        
        # Create optimizers for learner and adversary
        optimizer_learn = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.config["lr"], **self.hparams.opt_kwargs)
        optimizer_adv = self.hparams.optimizer(self.adversary.parameters(), lr=self.hparams.config["lr"], **self.hparams.opt_kwargs)

        return [optimizer_learn, optimizer_adv], []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.learner(x)
    
    
class Learner(nn.Module):
    def __init__(self, 
        num_features: int,
        hidden_units: List[int] = [64,32]
        ):        
        '''
        num_features - int, number of features of the input
        hidden_units - list, number of hidden units in each layer of the MLP
        '''
            
        super().__init__()
        
        # construct network
        net_list: List[torch.nn.Module] = []
        num_units = [num_features] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        num_features: int,
        hidden_units: List[int] = []
        ):
        
        '''
        num_features - int, number of features of the input
        hidden_units - list, number of hidden units in each layer of the MLP
        '''
            
        super().__init__()
        
        # construct network
        net_list: List[torch.nn.Module] = []
        num_units = [num_features] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))
        net_list.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_list)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
    
    
