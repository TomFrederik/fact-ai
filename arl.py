from typing import Dict, Type, Optional, Any, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models



class ARL(pl.LightningModule):
    """Feed forward neural network consisting of a primary network and an 
    adversary network that reweights the losses. 

    Attributes:
        config: Dict with hyperparameters (learning rate, batch size).
        input_shape: Dimensionality of the data input.
        pretrain_steps: Number of pretraining steps before using the DRO loss.
        prim_hidden: Number of hidden units in each layer of the primary network.
        adv_hidden: Number of hidden units in each layer of the adversary network.
        include_labels: Option to include labels as data inputs to the adversary
            network.
        optimizer: Optimizer used to update the model parameters.
        dataset_type: Type of the dataset; 'tabular' or 'image'.
        pretrained: Option to use a pretrained model if the networks are CNNs.
        opt_kwargs: Optional; optimizer keywords other than learning rate.
        
    Raises:
        Exception: If the dataset type is neither tabular nor image data.
    """

    def __init__(self, 
        config: Dict[str, Any],
        input_shape: int,
        pretrain_steps: int,
        prim_hidden: List[int] = [64,32],
        adv_hidden: List[int] = [],
        include_labels: bool = True,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
        dataset_type: str = 'tabular',
        pretrained: bool = False,
        opt_kwargs: Dict[str, Any] = {},
        ):
        """Inits an instance of ARL with the given attributes."""
        
        super().__init__()

        # save params
        self.save_hyperparameters()
        
        # init networks
        if dataset_type == 'tabular':
            self.learner = Learner(input_shape=input_shape, hidden_units=prim_hidden)
            self.adversary = Adversary(input_shape=input_shape, hidden_units=adv_hidden, include_labels=include_labels)
        elif dataset_type == 'image':
            # only works with [3, 224, 224] images since input shape of fully connected layers must be hard-coded
            assert input_shape == [3, 224, 224], f"Input shape to ARL is {input_shape} and not [3, 224, 224]!"
            self.learner = CNN_Learner(input_shape=input_shape, hidden_units=prim_hidden, pretrained=pretrained)
            self.adversary = CNN_Adversary(input_shape=input_shape, hidden_units=adv_hidden, pretrained=pretrained)
        else:
            raise Exception("ARL was unable to recognize the dataset type.")

        # init loss function
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    
    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int,
                      optimizer_idx: int) -> Optional[torch.Tensor]:
        """Computes and logs the adversarially reweighted loss on the training 
        set.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
            optimizer_idx: Index of the optimizer that is used for updating the 
                weights after the training step; 0 = learner, 1 = adversary.
    
        Returns:
            Adversarially reweighted loss or negative adversarially reweighted
            loss. During pretraining, only return the positive loss.
        """
        
        x, y, s = batch         

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
        """Computes the adversarially reweighted loss on the training set.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
            y: Tensor of shape [batch_size] with labels.
    
        Returns:
            Adversarially reweighted loss on the training dataset.
        """
        
        # compute unweighted bce
        logits = self.learner(x)        
        bce = self.loss_fct(logits, y)
        
        # compute lambdas
        lambdas = self.adversary(x, y)
        
        # compute reweighted loss  
        loss = torch.mean(lambdas * bce)
        
        return loss
     
        
    def adversary_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the negative adversarially reweighted loss on the training set.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
            y: Tensor of shape [batch_size] with labels.
    
        Returns:
            Negative adversarially reweighted loss on the training dataset.
        """
        
        # compute unweighted bce
        logits = self.learner(x)        
        bce = self.loss_fct(logits, y)
        
        # compute lambdas
        lambdas = self.adversary(x, y)
        
        # compute reweighted loss
        loss = -torch.mean(lambdas * bce)
        
        return loss        
        
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Computes and logs the adversarially reweighted loss on the validation 
        set.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        """
        
        x, y, s = batch        
        loss = self.learner_step(x, y)
        
        # logging
        self.log("validation/reweighted_loss_learner", loss)

        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Computes and logs the adversarially reweighted loss on the test set.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        """
        
        x, y, s = batch 
        loss = self.learner_step(x, y)
        
        # logging
        self.log("test/reweighted_loss_learner", loss)
 
    
    def configure_optimizers(self):
        """Chooses optimizers and learning-rates to use during optimization of
        the primary and adversary network.
        
        Returns:
            Optimizers.   
            Learning-rate schedulers (currently not used).
        """
        
        optimizer_learn = self.hparams.optimizer(self.learner.parameters(), lr=self.hparams.config["lr"], **self.hparams.opt_kwargs)
        optimizer_adv = self.hparams.optimizer(self.adversary.parameters(), lr=self.hparams.config["lr"], **self.hparams.opt_kwargs)

        return [optimizer_learn, optimizer_adv], []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation of inputs through the primary network.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """

        return self.learner(x)
    
    
class Learner(nn.Module):
    """Fully-connected feed forward neural network; primary network of the ARL. 

    Attributes:
        input_shape: Dimensionality of the data input.
        hidden_units: Number of hidden units in each layer of the network.
    """
    
    def __init__(self, 
        input_shape: int,
        hidden_units: List[int] = [64,32]
        ):        
        """Inits an instance of the primary network with the given attributes."""
            
        super().__init__()
        
        # construct network
        net_list: List[torch.nn.Module] = []
        num_units = [input_shape] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net = nn.Sequential(*net_list)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation of inputs through the primary network.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """
        
        out = self.net(x)
        
        return torch.squeeze(out, dim=-1)

    
class Adversary(nn.Module):
    """Fully-connected feed forward neural network; adversary network of the ARL. 

    Attributes:
        input_shape: Dimensionality of the data input.
        hidden_units: Number of hidden units in each layer of the network.
        include_labels: Option to include labels as data inputs to the network;
            defaults to True.
    """
    
    def __init__(self, 
        input_shape: int,
        hidden_units: List[int] = [],
        include_labels: bool = True
        ):
        """Inits an instance of the adversary network with the given attributes."""
            
        super().__init__()
        
        # construct network
        net_list: List[torch.nn.Module] = []
        if include_labels:
            # the adversary also takes the label as input, therefore we need the +1
            input_shape += 1
        self.include_labels = include_labels

        num_units = [input_shape] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))
        net_list.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_list)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward propagation of inputs and labels (optional) through the 
        adversary network.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
            y: Tensor of shape [batch_size] with labels.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """
        
        if self.include_labels:
            input = torch.cat([x, y.unsqueeze(1).float()], dim=1).float()
        else:
            input = x
        
        # compute adversary
        adv = self.net(input)
        
        # normalize adversary across batch
        # TODO: check numerical stability
        adv_norm = adv / torch.sum(adv)
        
        # scale and shift
        out = x.shape[0] * adv_norm + torch.ones_like(adv_norm)        
        
        return torch.squeeze(out, dim=-1)


class CNN_Learner(nn.Module):
    """Feed forward CNN (ResNet34); primary network of the ARL. 

    Attributes:
        input_shape: Dimensionality of the data input.
        hidden_units: Number of hidden units in each fully-connected layer of 
            the network.
        pretrained: Option to use a model that is pretrained on ImageNet.
    """
    
    def __init__(self,
                 input_shape: int,
                 hidden_units: list = [512, 32],
                 pretrained: bool = False
                 ):
        """Inits an instance of the primary CNN with the given attributes."""

        super().__init__()

        # construct network
        self.net = models.resnet34(pretrained=pretrained)

        net_list = []
        num_units = [512] + hidden_units
        for i in range(len(num_units) - 1):
            net_list.append(nn.Linear(num_units[i], num_units[i + 1]))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.net.fc = nn.Sequential(*net_list)

    def forward(self, x):
        """Forward propagation of inputs through the primary network.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """
        print(x.shape)
        test = self.net.conv1(x)
        out = self.net(x)

        return torch.squeeze(out, dim=-1)


class CNN_Adversary(nn.Module):
    """Feed forward CNN (ResNet34); adversary network of the ARL. 

    Attributes:
        input_shape: Dimensionality of the data input.
        hidden_units: Number of hidden units in each fully-connected layer of 
            the network.
        pretrained: Option to use a model that is pretrained on ImageNet.
    """
    
    def __init__(self,
                 input_shape: int,
                 hidden_units: list = [],
                 pretrained: bool = False
                 ):
        """Inits an instance of the adversary CNN with the given attributes."""

        super().__init__()

        # construct network and set default fc to identity for appending sample label during forward pass
        self.net = models.resnet34(pretrained=pretrained)
        self.net.fc = nn.Identity()

        net_list = []
        num_units = [512 + 1] + hidden_units
        for i in range(len(num_units) - 1):
            net_list.append(nn.Linear(num_units[i], num_units[i + 1]))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.fc = nn.Sequential(*net_list)

    def forward(self, x, y):
        """Forward propagation of inputs and labels (optional) through the 
        adversary network.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
            y: Tensor of shape [batch_size] with labels.
    
        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """

        # compute adversary
        intermediate = self.net(x)
        intermediate = torch.cat([intermediate.float(), y.float().unsqueeze(1)], dim=1)
        adv = self.fc(intermediate)

        # normalize adversary across batch
        # TODO: check numerical stability
        adv_norm = adv / torch.sum(adv)

        # scale and shift
        out = x.shape[0] * adv_norm + torch.ones_like(adv_norm)

        return torch.squeeze(out)
