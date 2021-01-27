from typing import Dict, Type, Optional, Any, List, Tuple, Set
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models # type: ignore



class ARL(pl.LightningModule):
    """Feed forward neural network consisting of a primary network and an 
    adversary network that reweights the losses. 

    Attributes:
        config: Dict with hyperparameters (learning rate, batch size).
        input_shape: Dimensionality of the data input.
        pretrain_steps: Number of pretraining steps before using the DRO loss.
        prim_hidden: Number of hidden units in each layer of the primary network.
        adv_hidden: Number of hidden units in each layer of the adversary network.
        adv_input: Set with strings describing the input the adversary has access to.
            May contain any combination of 'X' for features, 'Y' for ground truth labels and 'S'
            for protected class memberships. E.g. {'X', 'Y'} for the usual ARL method.
            If 'S' is set, num_groups must be the set as well. This set must not be empty
            (that would correspond to a constant adversary output, use Baseline instead).
            Currently only has an effect if dataset_type is 'tabular'.
        num_groups: Number of protected groups. Only needs to be set when
            adversary has access to protected group memberships.
        optimizer: Optimizer used to update the model parameters.
        dataset_type: Type of the dataset; 'tabular' or 'image'.
        pretrained: Option to use a pretrained model if the networks are CNNs.
        opt_kwargs: Optional; optimizer keywords other than learning rate.
        
    Raises:
        Exception: If the dataset type is neither tabular nor image data.
    """

    learner: nn.Module
    adversary: nn.Module

    def __init__(self, 
        config: Dict[str, Any],
        input_shape: int,
        pretrain_steps: int,
        prim_hidden: List[int] = [64,32],
        adv_hidden: List[int] = [],
        adv_input: Set[str] = {'X', 'Y'},
        num_groups: Optional[int] = None,
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
            self.adversary = Adversary(input_shape=input_shape, hidden_units=adv_hidden,
                                       adv_input=adv_input,
                                       num_groups=num_groups)
        elif dataset_type == 'image':
            if adv_input != {'X', 'Y'}:
                print('CNN architecture currently only supports X+Y as adversary input')
            # only works with (C: 1, H: 28, W: 28) images since input shape of fully connected layers must be hard-coded
            # assert input_shape == (1, 28, 28), f"Input shape to ARL is {input_shape} and not (1, 28, 28)!"
            self.learner = CNN_Learner(hidden_units=prim_hidden)
            self.adversary = CNN_Adversary(hidden_units=adv_hidden)
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
            loss = self.learner_step(x, y, s)
            
            # logging
            self.log("training/reweighted_loss_learner", loss)
            
            return loss

        elif optimizer_idx == 1 and self.global_step > self.hparams.pretrain_steps:
            loss = self.adversary_step(x, y, s)
            
            return loss

        else:
            return None


    def learner_step(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Computes the adversarially reweighted loss on the training set.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
            y: Tensor of shape [batch_size] with labels.
            s: Tensor of shape [batch_size] with protected group membership indices.
    
        Returns:
            Adversarially reweighted loss on the training dataset.
        """
        
        # compute unweighted bce
        logits = self.learner(x)        
        bce = self.loss_fct(logits, y)
        
        # compute lambdas
        lambdas = self.adversary(x, y, s)
        
        # compute reweighted loss  
        loss = torch.mean(lambdas * bce)
        
        return loss
     
        
    def adversary_step(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Computes the negative adversarially reweighted loss on the training set.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
            y: Tensor of shape [batch_size] with labels.
            s: Tensor of shape [batch_size] with protected group membership indices.
    
        Returns:
            Negative adversarially reweighted loss on the training dataset.
        """
        
        # compute unweighted bce
        
        logits = self.learner(x)        
        bce = self.loss_fct(logits, y)
        
        # compute lambdas
        lambdas = self.adversary(x, y, s)

        for i in range(lambdas.shape[0]):
            if lambdas[i] > 20:
                self.logger.experiment.add_image('high_lambda', x[i], 0)
        
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
        loss = self.learner_step(x, y, s)
        
        # logging
        self.log("validation/reweighted_loss_learner", loss)

        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Computes and logs the adversarially reweighted loss on the test set.
    
        Args:
            batch: Inputs, labels and group memberships of a data batch.
            batch_idx: Index of batch in the dataset (not needed).
        """
        
        x, y, s = batch 
        loss = self.learner_step(x, y, s)
        
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

    def get_lambda(self, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor]:
        """
        Evaluate the adversary on the given dataset

        Args:
            dataloader: An instance of torch.utils.data.DataLoader, e.g. the test dataloader
        
        Returns:
            lambdas: Tensor of shape [num_samples] containing the adversary scores lambda
            predictions: Tensor of shape [num_samples] containing predicted labels in the same order as the lambdas
            true_labels: Tensor of shape [num_samples] containing true labels in the same order as the lambdas
            memberships: Tensor of shape [num_samples] containing group memberships in the same order as the lambdas
        """

        lambdas = []
        true_labels = []
        predictions = []
        memberships = []

        for x, y, s in iter(dataloader):
            # put batch on correct device
            x = x.to(self.device)
            y = y.to(self.device)
            s = s.to(self.device)

            # put through adversary
            batch_lambdas = self.adversary(x, y, s)

            # put through learner
            batch_pred = torch.round(torch.sigmoid(self.learner(x)))

            # store results
            predictions.append(batch_pred)
            true_labels.append(y)
            lambdas.append(batch_lambdas)
            memberships.append(s)

        # cat tensors
        lambdas = torch.cat(lambdas, dim=0)
        predictions = torch.cat(predictions, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        memberships = torch.cat(memberships, dim=0)

        return lambdas, predictions, true_labels, memberships
    
    
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
        net_list: List[nn.Module] = []
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
        adv_input: Set with strings describing the input the adversary has access to.
            May contain any combination of 'X' for features, 'Y' for ground truth labels and 'S'
            for protected class memberships. E.g. {'X', 'Y'} for the usual ARL method.
            If 'S' is set, num_groups must be the set as well. Any strings other than 'X',
            'Y' and 'S' are ignored.
            This set must not be empty (that would correspond to a constant adversary output, use Baseline instead).
        num_groups: Number of protected groups. Only needs to be set if 'S' is used in adv_input.
    """
    
    def __init__(self, 
        input_shape: int,
        hidden_units: List[int] = [],
        adv_input: Set[str] = {'X', 'Y'},
        num_groups: Optional[int] = None,
        ):
        """Inits an instance of the adversary network with the given attributes."""
            
        super().__init__()

        if len(adv_input) == 0:
            raise ValueError("Adversary has no inputs!")
        
        # construct network
        net_list: List[nn.Module] = []
        num_inputs = 0
        if 'X' in adv_input:
            num_inputs += input_shape
        if 'Y' in adv_input:
            num_inputs += 1
        if 'S' in adv_input:
            assert num_groups is not None, "num_groups must be set when using protected features as input"
            num_inputs += num_groups
        self.adv_input = adv_input
        self.num_groups = num_groups

        num_units = [num_inputs] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))
        net_list.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_list)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward propagation of inputs and labels (optional) through the 
        adversary network.
    
        Args:
            x: Tensor of shape [batch_size, input_shape] with data inputs.
            y: Tensor of shape [batch_size] with labels.
            s: Tensor of shape [batch_size] with protected group membership indices.
    
        Returns:
            Tensor of shape [batch_size] with reweighting scores.
        """
        inputs: List[torch.Tensor] = []

        if 'X' in self.adv_input:
            inputs.append(x)
        if 'Y' in self.adv_input:
            inputs.append(y.unsqueeze(1).float())
        if 'S' in self.adv_input:
            inputs.append(nn.functional.one_hot(s.long(), num_classes=self.num_groups).float())
        
        input = torch.cat(inputs, dim=1).float()

        # compute adversary
        adv = self.net(input)
        
        # normalize adversary across batch
        # TODO: check numerical stability
        adv_norm = adv / torch.sum(adv)
        
        # scale and shift
        out = x.shape[0] * adv_norm + torch.ones_like(adv_norm)        
        
        return torch.squeeze(out, dim=-1)


class CNN_Learner(nn.Module):
    """Feed forward CNN; primary network of the ARL.

    Attributes:
        hidden_units: Number of hidden units in each fully-connected layer of
            the network.
    """

    def __init__(self,
                 hidden_units: list = [64, 32]
                 ):
        """Inits an instance of the primary CNN with the given attributes."""

        super().__init__()

        # construct network
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3)),
                                 nn.MaxPool2d(kernel_size=(2, 2)),
                                 nn.Flatten())
        net_list: List[torch.nn.Module] = []
        num_units = [10816] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))

        self.fc = nn.Sequential(*net_list)

    def forward(self, x):
        """Forward propagation of inputs through the primary network.

        Args:
            x: Tensor of shape [batch_size, 1, 28, 28] with data inputs.

        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """
        intermediate = self.cnn(x)
        out = self.fc(intermediate)

        return torch.squeeze(out, dim=-1)


class CNN_Adversary(nn.Module):
    """Feed forward CNN; adversary network of the ARL.

    Attributes:
        hidden_units: Number of hidden units in each fully-connected layer of
            the network.
    """

    def __init__(self,
                 hidden_units: list = []
                 ):
        """Inits an instance of the adversary CNN with the given attributes."""

        super().__init__()

        # construct network
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3)),
                                 nn.MaxPool2d(kernel_size=(2, 2)),
                                 nn.Flatten())
        net_list: List[torch.nn.Module] = []
        num_units = [10816 + 1] + hidden_units
        for num_in, num_out in zip(num_units[:-1], num_units[1:]):
            net_list.append(nn.Linear(num_in, num_out))
            net_list.append(nn.ReLU())
        net_list.append(nn.Linear(num_units[-1], 1))
        net_list.append(nn.Sigmoid())

        self.fc = nn.Sequential(*net_list)

    def forward(self, x, y, s):
        """Forward propagation of inputs and labels (optional) through the
        adversary network. Concatenates y to flattened output of CNN part.

        Args:
            x: Tensor of shape [batch_size, 1, 28, 28] with data inputs.
            y: Tensor of shape [batch_size] with labels.
            s: Tensor of shape [batch_size] with protected group membership indices (unused).

        Returns:
            Tensor of shape [batch_size] with predicted logits.
        """

        # compute adversary
        intermediate = self.cnn(x)
        intermediate = torch.cat([intermediate.float(), y.float().unsqueeze(1)], dim=1)
        adv = self.fc(intermediate)

        # normalize adversary across batch
        # TODO: check numerical stability
        adv_norm = adv / torch.sum(adv)

        # scale and shift
        out = x.shape[0] * adv_norm + torch.ones_like(adv_norm)

        return torch.squeeze(out)
