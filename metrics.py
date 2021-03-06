from typing import Dict, Type, Optional, Any, List, Tuple
from statistics import mean

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import DataLoader
from time import time

from datasets import FairnessDataset


class Logger(Callback):
    """Callback that logs various metrics.
    
    Logs micro-average AUC, macro-average AUC, minimum protected group AUC, AUC 
    of the protected group with the fewest members and accuracy.

    Attributes:
        dataset: Dataset instance to use. 
        name: Directory of the logged metrics, e.g. "training", "validation" 
            or "test".
        batch_size: Batch size to iterate through the dataset.
        save_scatter: Whether scatter plots of BCE loss vs lambdas should be saved (only for ARL)
    """
    
    def __init__(self, dataset: FairnessDataset, name: str, batch_size: int, save_scatter: bool = False):
        """Inits an instance of Logger with the given attributes."""
        
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.name = name
        self.save_scatter = save_scatter
        # create a dataloader to pass the dataset through the model
        self.dataloader = DataLoader(self.dataset, self.batch_size, pin_memory=True)

        if self.save_scatter:
            self.scatter_dataloader = DataLoader(self.dataset, 256, shuffle=True, pin_memory=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs metrics. Function is called at the end of each validation epoch.
    
        Args:
            trainer: Trainer instance that handles the training loop.
            pl_module: Model to evaluate the metrics on, e.g. an instance of 
                baseline, ARL, DRO or IPW.    
        """
        if self.name != 'train':
            super().on_validation_end(trainer, pl_module)

            results = get_all_auc_scores(pl_module, self.dataloader, self.dataset.minority)

            for key in results:
                pl_module.log(f'{self.name}/{key}', results[key])

        if self.save_scatter:
            save_scatter(pl_module, self.scatter_dataloader, self.name)


def get_all_auc_scores(pl_module: LightningModule, dataloader: DataLoader, minority: int) -> Dict[str, float]:
    """Computes different AUC scores and the accuracy of the given module on
    the given dataset.

    Args:
        pl_module: Model to evaluate the metrics on. 
        dataloader: Dataloader instance used to pass the dataset through the 
            model.
        minority: Index of the protected group with the fewest members.

    Returns:
        A dict mapping keys to the corresponding metric values.
    """
    
    # iterate through dataloader to generate predictions
    predictions: List[torch.Tensor] = []
    memberships: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for x, y, s in iter(dataloader):
        x = x.to(pl_module.device)
        # y and s are simple scalars, no need to move to GPU
        batch_predictions = torch.sigmoid(pl_module(x))
        predictions.append(batch_predictions)
        memberships.append(s)
        targets.append(y)

    prediction_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0).to(prediction_tensor.device)
    membership_tensor = torch.cat(memberships, dim=0).to(prediction_tensor.device)

    aucs = group_aucs(prediction_tensor, target_tensor, membership_tensor)
    acc = torch.mean(((prediction_tensor > 0.5).int() == target_tensor).float()).item()
    
    results = {
        'min_auc': min(aucs.values()),
        'macro_avg_auc': mean(aucs.values()),
        'micro_avg_auc': auroc(prediction_tensor, target_tensor).item(),
        'minority_auc': aucs[minority],
        'accuracy': acc
    }
    
    return results


def group_aucs(predictions: torch.Tensor, targets: torch.Tensor, memberships: torch.Tensor) -> Dict[int, float]:
    """Computes the AUC for each protected group.

    Args:
        predictions: Tensor of shape (n_samples, ) with prediction logits.
        targets: Tensor of shape (n_samples, ) with ground truth (0 or 1).
        memberships: Tensor of shape (n_samples, ) with group membership indices.

    Returns:
        A dict mapping group indices as keys to the corresponding AUC values.        
    """
    
    groups = memberships.unique().to(predictions.device)
    groups = groups.to(predictions.device)
    targets = targets.to(predictions.device)
    memberships = memberships.to(predictions.device)
    aucs: Dict[int, float] = {}
    
    for group in groups:
        indices = (memberships == group)
        if torch.sum(targets[indices]) == 0 or torch.sum(1-targets[indices]) == 0:
            aucs[int(group)] = 0 
        else:
            aucs[int(group)] = auroc(predictions[indices], targets[indices]).item()
    return aucs


def save_scatter(pl_module: LightningModule, dataloader: DataLoader, name: str):
    """Calls the save_scatter method of the ARL module for a mini batch.

    Args:
        pl_module: LightningModule of the model.
        dataloader: Dataloader of the Logger instance dataset
        name: Name that will be added to the plot name
    """
    x, y, s = next(iter(dataloader))
    pl_module.save_scatter(x.to(pl_module.device), y.to(pl_module.device), s.to(pl_module.device), name)
