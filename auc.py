from statistics import mean

from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.callbacks import Callback


class AUCLogger(Callback):
    def __init__(self, dataset):
        """Callback that logs various AUC metrics.

        Args:
            dataset: Dataset instance to use"""
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        scores = pl_module(self.dataset.features)
        aucs = aucs_from_dataset(scores, self.dataset)
        pl_module.log("validation/min_auc", min(aucs.values()))
        pl_module.log("validation/avg_auc", mean(aucs.values()))
        pl_module.log("validation/minority_auc", aucs[self.dataset.minority])


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
