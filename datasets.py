from typing import Dict, Type, Optional, Any, List, Tuple, Union
from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import json
import itertools
from torchvision import transforms, datasets  # type: ignore
from PIL import Image # type: ignore

DATASET_SETTINGS: Dict[str, Dict[str, Any]] = {
    "Adult": {
        "sensitive_column_names": ['race', 'sex'],
        "sensitive_column_values": ['Black', 'Female'],
        "target_variable": "income",
        "target_value": ">50K"},
    "LSAC": {
        "sensitive_column_names": ['race', 'sex'],
        "sensitive_column_values": ['Black', 'Female'],
        "target_variable": "pass_bar",
        "target_value": "Passed"
    },
    "COMPAS": {
        "sensitive_column_names": ['race', 'sex'],
        "sensitive_column_values": ['Black', 'Female'],
        "target_variable": "is_recid",
        "target_value": "Yes"
    }
}


class FairnessDataset(ABC, Dataset):
    """Abstract base class used for TabularDataset."""
    
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, float, int]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def protected_index2value(self) -> Dict[int, Any]:
        pass

    @property
    @abstractmethod
    def features(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def dimensionality(self) -> int:
        pass

    @property
    @abstractmethod
    def minority(self) -> int:
        pass

    @property
    @abstractmethod
    def group_probs(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def memberships(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def labels(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def sensitive_label(self) -> bool:
        pass


class TabularDataset(FairnessDataset):
    """Dataset from tabular data that can provide information about protected
    groups amongst its elements. 

    Attributes:
        dataset_name: Identifier of the dataset used to load the data and the
            corresponding dataset settings.
        test: Option to use the test dataset.
        hide_sensitive_columns: Option to remove sensitive columns from the 
            dataset.
        binarize_prot_group: Option to binarize the values in sensitive columns.
            If true, the dataset will only differentiate between sensitive and 
            non-sensitive values (e.g. 'black' and 'not black') in sensitive 
            columns.
        idcs: Optional; indices that specify which rows should be included in 
            the dataset. If None, all rows are included.
        sensitive_label: Option to use the joint probability of label and group
            membership for computing the weights for the IPW (IPW(S+Y)).
        disable_warnings: Option to show warnings if mean or std of the dataset
            exceed a certain threshold after normalization.
        suffix: Option to specify suffix of dataset files
    """

    def __init__(self, dataset_name: str,
                 test: bool = False,
                 hide_sensitive_columns: bool = True,
                 binarize_prot_group: bool = True,
                 idcs: Optional[List[int]] = None,
                 sensitive_label: bool = False,
                 disable_warnings: bool = False,
                 suffix: str = ''):

        super().__init__()
        """Inits an instance of TabularDataset with the given attributes."""

        base_path = os.path.join("data", dataset_name)
        vocab_path = os.path.join(base_path, "vocabulary.json")
        path = os.path.join(base_path, f"test{suffix}.csv" if test else f"train{suffix}.csv")
        mean_std_path = os.path.join(base_path, f"mean_std{suffix}.json")
        sensitive_column_names = DATASET_SETTINGS[dataset_name]["sensitive_column_names"].copy()
        sensitive_column_values = DATASET_SETTINGS[dataset_name]["sensitive_column_values"].copy()
        target_variable = DATASET_SETTINGS[dataset_name]["target_variable"]
        target_value = DATASET_SETTINGS[dataset_name]["target_value"]

        self.hide_sensitive_columns = hide_sensitive_columns
        self._sensitive_label = sensitive_label

        # load data
        features = pd.read_csv(path, ',', header=0)
        columns: List[str] = list(features.columns)

        # drop indices not specified
        if idcs is not None:
            features = features.iloc[idcs]

        # load mean and std
        with open(mean_std_path) as json_file:
            mean_std: Dict[str, Tuple[float, float]] = json.load(json_file)

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

        # create labels
        labels = (features[target_variable].to_numpy() == target_value).astype(int)
        self._labels: torch.Tensor = torch.from_numpy(labels)


        # if set, will binarize/group values in the sensitive columns
        if binarize_prot_group:
            for col, val in zip(sensitive_column_names, sensitive_column_values):
                features[col] = features[col].apply(lambda x: float(x == val))

        # turn protected group memberships into a single index
        # first create lists of all the values the sensitive columns can take:
        uniques = [tuple(features[col].unique()) for col in sensitive_column_names]
        # create a list of tuples of such values. This corresponds to a list of all protected groups
        index2values = list(itertools.product(*uniques))
        # create the inverse dictionary:
        values2index = {vals: index for index, vals in enumerate(index2values)}
        if binarize_prot_group:
            # We want a dictionary that assigns each protected group index
            # (e.g. 0, 1, 2, 3) to its meaning (e.g. ("Black", "Male")).
            # If we have binarized protected groups, this isn't possible
            # but we can at least get descriptions such as ("Black", "Other")
            # i.e. the sensitive value or else "Other".
            self.index2values = {
                index: tuple(
                    val if vals[i] == 1 else "Other"
                    for i, val in enumerate(sensitive_column_values)
                ) for index, vals in enumerate(index2values)}
        else:
            self.index2values = {i: val for i, val in enumerate(index2values)}

        # remove target variable from features
        columns.remove(target_variable)
        self.sensitives = features[sensitive_column_names]
        if hide_sensitive_columns:  # remove sensitive columns
            for c in sensitive_column_names:
                columns.remove(c)
        features = features[columns]

        # Create a tensor with protected group membership indices for easier access
        # Ignore type because mypy throws an error even though this is correct
        self._memberships = torch.empty(len(features), dtype=int) # type: ignore
        for i in range(len(self.memberships)):
            s = tuple(self.sensitives.iloc[i])
            self._memberships[i] = values2index[s]

        # compute the minority group (the one with the fewest members) and group probabilities
        vals, counts = self.memberships.unique(return_counts=True)
        self._minority = vals[counts.argmin().item()].item()

        # calculate group probabilities for IPW
        if sensitive_label:
            prob_identifier = torch.stack([self.memberships, self.labels], dim=1)
            vals, counts = prob_identifier.unique(return_counts=True, dim=0)
            probs = torch.true_divide(counts, torch.sum(counts))
            self._group_probs = probs.reshape(-1, 2)
        else:
            vals, counts = self.memberships.unique(return_counts=True)
            self._group_probs = torch.true_divide(counts, torch.sum(counts).float())
            
        ## convert categorical data into onehot
        # load vocab
        with open(vocab_path) as json_file:
            vocab = json.load(json_file)

        # we already mapped target var values to 0 and 1 before
        del vocab[target_variable]

        tensors: List[torch.Tensor] = []
        for c in columns:
            if c in vocab:
                vals = list(vocab[c])
                val2int = {vals[i]: i for i in range(len(vals))}  # map possible value to integer
                features[c] = features[c].apply(lambda x: val2int[x])
                one_hot = nn.functional.one_hot(
                    torch.tensor(features[c].values).long(),
                    len(vals))
                for i in range(one_hot.size(-1)):
                    tensors.append(one_hot[:, i].float())
            else:
                tensors.append(torch.tensor(features[c].values).float())

        self._features = torch.stack(tensors, dim=1).float()

    def __len__(self):
        """Returns the number of elements in the dataset."""
        return self.features.size(0)

    def __getitem__(self, index):
        """Returns specified elements of the dataset.
        
        Args:
            index: Indices of elements to return.
            
        Returns:
            x: Features of the specified elements.
            y: Labels of the specified elements.
            s: Group memberships of the specified elements.       
        """

        x = self.features[index]

        y = float(self.labels[index])

        s = self.memberships[index].item()

        return x, y, s

    @property
    def protected_index2value(self):
        """List that turns the index of a protected group into meaningful values.

        Dataset.protected_index2value[index] turns the index of a protected group
        into a tuple such as ("White", "Male") that specifies the values of the 
        underlying sensitive attributes."""
        return self.index2values

    @property
    def features(self):
        """Features of all elements of the dataset."""
        return self._features

    @property
    def dimensionality(self):
        """Dimensionality of single dataset elements."""
        return self.features.size(1)

    @property
    def minority(self):
        """Index of the protected group with the fewest members."""
        return self._minority

    @property
    def group_probs(self):
        """Empirical observation probabilities of the protected groups."""
        return self._group_probs

    @property
    def memberships(self):
        """Group memberships of all elements of the dataset."""
        return self._memberships

    @property
    def labels(self):
        """Labels of all elements of the dataset."""
        return self._labels

    @property
    def sensitive_label(self):
        """Whether the label should be included in IPW"""
        return self._sensitive_label


class EMNISTDataset(FairnessDataset):
    """Dataset from image data that can provide information about protected
    groups amongst its elements. 

    Attributes:
        noise: Whether the dataset should include noisy samples or not.
        imb: Whether to use an imbalanced dataset.
        test: Option to use the test dataset.
        idcs: Optional; indices that specify which rows should be included in
            the dataset. If None, all rows are included.
    """

    def __init__(self,
                 noise: bool = True,
                 imb: bool = False,
                 test: bool = False,
                 idcs: Optional[List[int]] = None):
        """Inits an instance of FairFaceDataset with the given attributes."""

        super().__init__()

        self._sensitive_label = False
        self.test = test
        self.to_tensor = transforms.ToTensor()

        if imb:
            dataset_name = 'EMNIST_10'
        else:
            dataset_name = "EMNIST_35"

        if self.test:
            self._data = np.load(os.path.join('data', dataset_name, 'test_prepared.npy'), allow_pickle=True)
        else:
            self._data = np.load(os.path.join('data', dataset_name, 'train_prepared.npy'), allow_pickle=True)

        if idcs is not None:
            self._data = self._data[idcs]

        self._dimensionality = np.expand_dims(np.array(self._data[0, 0]), axis=0).shape
        self._features = torch.stack([torch.Tensor(d[0] / 255).float().unsqueeze(0) for d in self._data])
        self.index2values = {0: 'unprotected', 1: 'protected'}
        protected_prob = np.mean([d[2] for d in self._data])
        self._group_probs = np.array([1 - protected_prob, protected_prob])
        self._memberships = torch.Tensor([d[2] for d in self._data])
        self._labels = torch.Tensor([d[1] for d in self._data]).float()

    def __len__(self):
        """Returns the number of elements in the dataset."""
        return len(self._data)

    def __getitem__(self, index):
        """Opens, converts and normalizes the images of specified elements and 
        returns the elements.
        
        Args:
            index: Indices of elements to return.
            
        Returns:
            x: Images of the specified elements.
            y: Labels of the specified elements.
            s: Group memberships of the specified elements.       
        """
        
        x = self._features[index]
        y = self._labels[index]
        s = self._memberships[index]

        return x, y, s

    @property
    def protected_index2value(self):
        """List that turns the index of a protected group into meaningful values.

        Dataset.protected_index2value[index] turns the index of a protected group
        into a tuple such as ("White", "Male") that specifies the values of the 
        underlying sensitive attributes."""
        return self.index2values

    @property
    def dimensionality(self):
        """Dimensionality of single images."""
        return self._dimensionality

    @property
    def minority(self):
        """Index of the protected group with the fewest members."""
        return 1

    @property
    def group_probs(self):
        """Empirical observation probabilities of the protected groups."""
        return self._group_probs

    @property
    def memberships(self):
        """Group memberships of all elements of the dataset."""
        return self._memberships

    @property
    def labels(self):
        """Labels of all elements of the dataset."""
        return self._labels

    @property
    def features(self):
        """Features of all elements of the dataset."""
        return self._features

    @property
    def sensitive_label(self):
        """Whether the label should be included in IPW"""
        return self._sensitive_label


class CustomSubset(FairnessDataset):
    """Subset of a dataset at specified indices.

    Arguments:
        dataset: The whole TabularDataset.
        indices: Indices in the whole set selected for subset.
    """

    def __init__(self, dataset: FairnessDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

        # calculate group probabilities for IPW
        if self.dataset.sensitive_label:
            prob_identifier = torch.stack([self.memberships, self.labels], dim=1)
            vals, counts = prob_identifier.unique(return_counts=True, dim=0)
            probs = torch.true_divide(counts, torch.sum(counts))
            self._group_probs = probs.reshape(-1, 2)
        else:
            vals, counts = self.memberships.unique(return_counts=True)
            self._group_probs = torch.true_divide(counts, torch.sum(counts).float())

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
        return self._group_probs

    @property
    def memberships(self):
        return self.dataset.memberships[self.indices]

    @property
    def labels(self):
        return self.dataset.labels[self.indices]
