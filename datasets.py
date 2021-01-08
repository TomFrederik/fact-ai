import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import itertools


DATASET_SETTINGS = {"Adult": {
        "vocab_path": "./data/uci_adult/vocabulary.json",
        "train_path": "./data/uci_adult/train.csv",
        "test_path": "./data/uci_adult/test.csv",
        'mean_std_path':'./data/uci_adult/mean_std.json',
        "columns": ["age", "workclass", "fnlwgt", "education", "education-num",
                    "marital-status", "occupation", "relationship", "race",
                    "sex", "capital-gain", "capital-loss", "hours-per-week",
                    "native-country", "income"],
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['Black','Female'],
        "target_variable": "income",
        "target_value": " >50K"},
    "LSAC": {
        'vocab_path':"./data/law_school/vocabulary.json",
        "train_path": "./data/law_school/train.csv",
        "test_path": "./data/law_school/test.csv",
        'mean_std_path':'./data/law_school/mean_std.json',
        'columns':["zfygpa",
                    "zgpa", 
                    "DOB_yr",
                    "isPartTime",
                    "sex",  
                    "race", 
                    "cluster_tier", 
                    "family_income", 
                    "lsat",
                    "ugpa",
                    "pass_bar",
                    "weighted_lsat_ugpa"
                ],
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['Black','Female'],
        "target_variable": "pass_bar",
        "target_value": "Passed"
        },
    "COMPAS":{
        'vocab_path':"./data/compas/vocabulary.json",
        "train_path": "./data/compas/train.csv",
        "test_path": "./data/compas/test.csv",
        'mean_std_path':'./data/compas/mean_std.json',
        'columns':["juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
                    "age", "c_charge_degree", "c_charge_desc", "age_cat", "sex", "race",
                    "is_recid"],
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['Black','Female'],
        "target_variable": "is_recid",
        "target_value": "Yes"
        }
    }


class Dataset(Dataset):

    def __init__(self, dataset_name, test=False, hide_sensitive_columns=True, binarize_prot_group=True):
        '''
        dataset_name - str, identifier of the dataset
        test - bool, whether to use the test set
        hide_sensitive_columns - bool, whether to hide (delete) sensitive columns
        binarize_prot_group - bool, whether to binarize the protected group. If true, all races other than black will be mapped to 0.
                                    If false, a unique index for each combination of sensitive column values is created.
        '''
        super().__init__()

        vocab_path = DATASET_SETTINGS[dataset_name]["vocab_path"]
        path = DATASET_SETTINGS[dataset_name]["test_path" if test else "train_path"]
        mean_std_path = DATASET_SETTINGS[dataset_name]['mean_std_path']
        columns = DATASET_SETTINGS[dataset_name]["columns"].copy()
        sensitive_column_names = DATASET_SETTINGS[dataset_name]["sensitive_column_names"].copy()
        sensitive_column_values = DATASET_SETTINGS[dataset_name]["sensitive_column_values"].copy()
        target_variable = DATASET_SETTINGS[dataset_name]["target_variable"]
        target_value = DATASET_SETTINGS[dataset_name]["target_value"]

        self.hide_sensitive_columns = hide_sensitive_columns

        # load data
        features = pd.read_csv(path, ',', names=columns)

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
                assert np.abs(np.mean(features[key])) < 1e-10
                assert np.abs(np.std(features[key]) - 1) < 1e-4

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
        print(self.values2index)

        # remove target variable from features
        columns.remove(target_variable)
        self.sensitives = features[sensitive_column_names]
        if hide_sensitive_columns: # remove sensitive columns
            for c in sensitive_column_names:
                columns.remove(c)
        features = features[columns]

        # Create a tensor with protected group membership indices for easier access
        self.memberships = torch.empty(len(features), dtype=int)
        for i in range(len(self.memberships)):
            s = tuple(self.sensitives.iloc[i])
            self.memberships[i] = self.values2index[s]

        # compute the minority group (the one with the fewest members)
        vals, counts = self.memberships.unique(return_counts=True)
        self.minority = vals[counts.argmin().item()].item()

        ## convert categorical data into onehot
        # load vocab
        with open(vocab_path) as json_file:
            vocab = json.load(json_file)
        
        # we already mapped target var values to 0 and 1 before
        del vocab[target_variable]
        

        # fill nan values in COMPAS dataset
        if dataset_name == 'COMPAS':
            features['c_charge_desc'].fillna('nan', inplace=True)

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


        # store dimensionality of x
        #x_ = list(features.iloc[0].to_numpy())
        #self.dimensionality = torch.flatten(torch.cat(x_, dim=0)).shape[0]

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


if __name__ == '__main__':

    # adult_dataset = Dataset("Adult")
    # print('\n\nExample 1 of Adult set: \n',adult_dataset[1])
    # print(adult_dataset.features["age"].unique())

    # compas_dataset = Dataset('COMPAS')
    # print('\n\nExample 1 of COMPAS set: \n',compas_dataset[1])

    lsac_dataset = Dataset('LSAC')
    print('\n\nExample 1 of LSAC set: \n', lsac_dataset[1])
