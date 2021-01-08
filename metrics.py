from statistics import mean

from pytorch_lightning.metrics.functional.classification import auroc, accuracy
from pytorch_lightning.callbacks import Callback
import torch


class Logger(Callback):
    def __init__(self, dataset, name):
        """Callback that logs various AUC metrics.

        Args:
            dataset: Dataset instance to use
            name: directory of the logged metrics, e.g. test or training
        """
        super().__init__()
        self.dataset = dataset
        self.name = name

    def on_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        scores = torch.sigmoid(pl_module(self.dataset.features))
        aucs = aucs_from_dataset(scores, self.dataset)
        acc = accuracy(scores, self.dataset.labels).item()

        pl_module.log(f"{self.name}/min_auc", min(aucs.values()))
        pl_module.log(f"{self.name}/macro_avg_auc", mean(aucs.values()))
        pl_module.log(f"{self.name}/micro_avg_auc", auroc(scores, self.dataset.labels).item())
        pl_module.log(f"{self.name}/minority_auc", aucs[self.dataset.minority])
        pl_module.log(f"{self.name}/accuracy", acc)


def group_aucs(predictions, targets, memberships):
    """Compute the AUROC for each protected group.

    Args:
        predictions: tensor of shape (n_samples, ) with predictions (either probabilities or scores)
        targets: tensor with same shape with ground truth (0 or 1)
        memberships: tensor with same shape with group membership indices

    Returns:
        A dictionary with the group indices as keys and their AUROCs as values"""
    groups = memberships.unique()
    aucs = {}
    for group in groups:
        indices = (memberships == group)
        if torch.sum(targets[indices]) == 0 or torch.sum(1-targets[indices]) == 0:
            aucs[group.item()] = 0 
        else:
            aucs[group.item()] = auroc(predictions[indices], targets[indices]).item()

    return aucs


def aucs_from_dataset(predictions, dataset):
    """Compute the AUROC for each protected group in an entire dataset.

    Args:
        predictions: tensor of shape (n_samples, ) with predictions (either probabilities or scores)
        dataset: a Dataset instance with length n_samples

    Returns:
        A dictionary with the group indices as keys and their AUROCs as values"""
    return group_aucs(predictions, dataset.labels, dataset.memberships)
