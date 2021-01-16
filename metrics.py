from statistics import mean

from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import DataLoader
from time import time


class Logger(Callback):
    def __init__(self, dataset, name, batch_size):
        """Callback that logs various AUC metrics.

        Args:
            dataset: Dataset instance to use
            name: directory of the logged metrics, e.g. test or training
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.name = name
        # create a dataloader to pass the dataset through the model
        self.dataloader = DataLoader(self.dataset, self.batch_size, pin_memory=True)

    def on_epoch_end(self, trainer, pl_module):
        super().on_epoch_end(trainer, pl_module)


        #print(f'Logging {self.name} at epoch {pl_module.current_epoch}')

        auctime = time()
        results = get_all_auc_scores(pl_module, self.dataloader, self.dataset.minority)
        #print(f'computing auc for {self.name} took {time()-auctime} seconds')

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
    groups = memberships.unique().to(predictions.device)
    groups = groups.to(predictions.device)
    targets = targets.to(predictions.device)
    aucs = {}
    
    for group in groups:
        indices = (memberships == group)
        if torch.sum(targets[indices]) == 0 or torch.sum(1-targets[indices]) == 0:
            aucs[int(group)] = 0 
        else:
            aucs[int(group)] = auroc(predictions[indices], targets[indices]).item()
    return aucs

# deprecated
def aucs_from_dataset(predictions, dataset):
    """Compute the AUROC for each protected group in an entire dataset.

    Args:
        predictions: tensor of shape (n_samples, ) with predictions (either probabilities or scores)
        dataset: a Dataset instance with length n_samples

    Returns:
        A dictionary with the group indices as keys and their AUROCs as values"""
    return group_aucs(predictions, dataset.labels, dataset.memberships)


def get_all_auc_scores(pl_module, dataloader, minority):
    '''
    Computes all the different AUC scores of the given module on the given dataset
    '''
    # iterate through dataloader to generate predictions
    predictions = []
    memberships = []
    targets = []
    #avg_fw_time = 0
    #fw_time = time()
    #num_steps = 0
    with torch.no_grad():
        for x, y, s in iter(dataloader):
            #num_steps+=1
            x = x.to(pl_module.device)
            y = y.to(pl_module.device)
            s = s.to(pl_module.device)
        
            batch_fw_time = time()
        
            batch_predictions = torch.sigmoid(pl_module(x))
            #avg_fw_time += time() - batch_fw_time
            predictions.append(batch_predictions)
            memberships.append(s)
            targets.append(y)
        #print(f'fw time was {time() - fw_time}')
        #print(f'avg fw time was {avg_fw_time/num_steps}')
        
        #cattime = time()
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        memberships = torch.cat(memberships, dim=0)
        #print(f'cat time was {time()-cattime}')

        #print("predictions.shape = ", predictions.shape)
        #print("targets.shape = ", targets.shape)
        #print("memberships.shape = ", memberships.shape)
        
        #auctime = time()
        aucs = group_aucs(predictions, targets, memberships)
        #print(f'time to compute auc was {time() - auctime}')

        #acctime = time()
        acc = torch.sum(predictions == targets).item()
        #print(f'acc time was {time() - acctime}')
    
    results = {'min_auc':min(aucs.values()),
                'macro_avg_auc': mean(aucs.values()),
                'micro_avg_auc': auroc(predictions, targets).item(),
                'minority_auc': aucs[minority],
                'accuracy':acc
    }
    
    return results
    