from statistics import mean

from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.metrics.classification import Accuracy
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
        self.accuracy = Accuracy()

    def on_epoch_end(self, trainer, pl_module):
        super().on_epoch_end(trainer, pl_module)

        results = get_all_auc_scores(pl_module, self.dataset)
        
        for key in results:
            pl_module.log(f'{self.name}/{key}', results[key])


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


def get_all_auc_scores(pl_module, dataset):
    '''
    Computes all the different AUC scores of the given module on the given dataset
    '''
    accuracy = Accuracy()
    scores = torch.sigmoid(pl_module(dataset.features))
    aucs = aucs_from_dataset(scores, dataset)
    acc = accuracy(scores, dataset.labels).item()

    results = {'min_auc':min(aucs.values()),
                'macro_avg_auc': mean(aucs.values()),
                'micro_avg_auc': auroc(scores, dataset.labels).item(),
                'minority_auc': aucs[dataset.minority],
                'accuracy':acc
    }
    
    return results
    