import os
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import pandas as pd
import numpy as np
import json
import itertools


DATASET_SETTINGS = {"Adult": {
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['Black','Female'],
        "target_variable": "income",
        "target_value": ">50K"},
    "LSAC": {
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['Black','Female'],
        "target_variable": "pass_bar",
        "target_value": "Passed"
        },
    "COMPAS":{
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['Black','Female'],
        "target_variable": "is_recid",
        "target_value": "Yes"
        }
    }


class CustomDataset(Dataset):

    def __init__(self, dataset_name, test=False, hide_sensitive_columns=True, binarize_prot_group=True, idcs=None,
                 sensitive_label=False, disable_warnings=False):
        """
        Dataset class for creating a dataset
        :param dataset_name: str, identifier of the dataset
        :param test: bool, whether to use the test set
        :param hide_sensitive_columns: bool, whether to hide (delete) sensitive columns
        :param binarize_prot_group: bool, whether to binarize the protected group. If true, all races other than black will be mapped to 0.
                                    If false, a unique index for each combination of sensitive column values is created.
        :param idcs: list, indices indicating which elements to take
        :param sensitive_label: bool, whether to include the target variable as a protected feature
        :param disable_warnings: bool, whether to show warnings regarding mean and std of the dataset
        """
        super().__init__()

        base_path = os.path.join("data", dataset_name)
        vocab_path = os.path.join(base_path, "vocabulary.json")
        path = os.path.join(base_path, "test.csv" if test else "train.csv")
        mean_std_path = os.path.join(base_path, "mean_std.json")
        sensitive_column_names = DATASET_SETTINGS[dataset_name]["sensitive_column_names"].copy()
        sensitive_column_values = DATASET_SETTINGS[dataset_name]["sensitive_column_values"].copy()
        target_variable = DATASET_SETTINGS[dataset_name]["target_variable"]
        target_value = DATASET_SETTINGS[dataset_name]["target_value"]

        self.hide_sensitive_columns = hide_sensitive_columns

        # load data
        features = pd.read_csv(path, ',', header=0)
        columns = list(features.columns)

        # drop indices not specified
        if idcs is not None:
            features = features.iloc[idcs]

        # load mean and std
        with open(mean_std_path) as json_file:
            mean_std = json.load(json_file)

        # center and normalize numerical features
        for key in mean_std:
            features[key] -= mean_std[key][0]
            features[key] /= mean_std[key][1]

            if not test:
                # in the training set, features should be precisely normalized
                # in the test set, we expect slight deviations
                mean = np.abs(np.mean(features[key]))
                delta_std = np.abs(np.std(features[key]) - 1)
                if not disable_warnings:
                    if mean > 1e-10:
                        print(f'WARNING: mean is {mean}')
                    if delta_std > 1e-4:
                        print(f'WARNING: delta std is {delta_std}')
                #assert mean < 1e-10, f'mean is {mean}'
                #assert delta_std < 1e-4, f'delta std is {delta_std}'

        # create labels
        self.labels = (features[target_variable].to_numpy() == target_value).astype(int)
        self.labels = torch.from_numpy(self.labels)

        # if set, will binarize/group values in the sensitive columns
        if binarize_prot_group:
            for col, val in zip(sensitive_column_names, sensitive_column_values):
                features[col] = features[col].apply(lambda x: float(x == val))
                
        
        # turn protected group memberships into a single index
        # first create lists of all the values the sensitive columns can take:
        uniques = [tuple(features[col].unique()) for col in sensitive_column_names]
        # create a list of tuples of such values. This corresponds to a list of all protected groups
        self.index2values = itertools.product(*uniques)
        # create the inverse dictionary:
        self.values2index = {vals: index for index, vals in enumerate(self.index2values)}

        # remove target variable from features
        columns.remove(target_variable)
        self.sensitives = features[sensitive_column_names]
        if hide_sensitive_columns:  # remove sensitive columns
            for c in sensitive_column_names:
                columns.remove(c)
        features = features[columns]

        # Create a tensor with protected group membership indices for easier access
        self.memberships = torch.empty(len(features), dtype=int)
        for i in range(len(self.memberships)):
            s = tuple(self.sensitives.iloc[i])
            self.memberships[i] = self.values2index[s]

        # compute the minority group (the one with the fewest members) and group probabilities
        vals, counts = self.memberships.unique(return_counts=True)
        self.minority = vals[counts.argmin().item()].item()

        # calculate group probabilities for IPW
        if sensitive_label:
            prob_identifier = torch.stack([self.memberships, self.labels], dim=1)
            vals, counts = prob_identifier.unique(return_counts=True, dim=0)
            probs = counts / torch.sum(counts)
            self.group_probs = probs.reshape(-1,2)
        else:
            vals, counts = self.memberships.unique(return_counts=True)
            self.group_probs = counts / torch.sum(counts)

        ## convert categorical data into onehot
        # load vocab
        with open(vocab_path) as json_file:
            vocab = json.load(json_file)
        
        # we already mapped target var values to 0 and 1 before
        del vocab[target_variable]
        
        class_columns = []
        num_classes = []
        tensors = []
        for c in columns:
            if c in vocab:
                vals = list(vocab[c])
                val2int = {vals[i]:i for i in range(len(vals))} # map possible value to integer
                features[c] = features[c].apply(lambda x: val2int[x])
                one_hot = nn.functional.one_hot(
                    torch.tensor(features[c].values).long(),
                    len(vals))
                for i in range(one_hot.size(-1)):
                    tensors.append(one_hot[:, i])
            else:
                tensors.append(torch.tensor(features[c].values))

        self.features = torch.stack(tensors, dim=1).float()
        self.dimensionality = self.features.size(1)


    def __len__(self):
        return self.features.size(0)
    
    def __getitem__(self, index):
        x = self.features[index]

        y = float(self.labels[index])

        s = self.memberships[index].item()

        return x, y, s

    @property
    def protected_index2value(self):
        """List that turns the index of a protected group into meaningful values.

        Dataset.__getitem__() returns three values, the third of which is an integer
        index that specifies the protected group the item belongs to.
        With Dataset.protected_index2value[index] you can turn this into a tuple
        such as ("White", "Male") that specifies the values of the underlying
        sensitive attributes. The order of attributes is the same as in the
        `sensitive_column_names` argument from `DATASET_SETTINGS`."""
        return self.index2values

class CustomSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole CustomDataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    @property
    def protected_index2value(self):
        return self.dataset.protected_index2value
    
    @property
    def features(self):
        return self.dataset.features[self.indices]
    
    @property
    def dimensionality(self):
        return self.dataset.dimensionality
    
    @property
    def minority(self):
        return self.dataset.minority
    
    @property
    def group_probs(self):
        return self.dataset.group_probs
    
    @property
    def memberships(self):
        return self.dataset.memberships[self.indices]
    
    @property
    def labels(self):
        return self.dataset.labels[self.indices]


if __name__ == '__main__':

    adult_dataset = CustomDataset("Adult")
    print('\n\nExample 1 of Adult set: \n',adult_dataset[1])

    compas_dataset = CustomDataset('COMPAS')
    print('\n\nExample 1 of COMPAS set: \n',compas_dataset[1])

    lsac_dataset = CustomDataset('LSAC')
    print('\n\nExample 1 of LSAC set: \n', lsac_dataset[1])
    #indices = [1,2,3,4]
    #subsetloader = DataLoader(lsac_dataset, batch_size=3, sampler=SubsetRandomSampler(indices))
    #print('\n\nFirst batch in subsetloader:\n',next(enumerate(subsetloader)))
