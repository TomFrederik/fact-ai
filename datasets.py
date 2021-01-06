import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import json
import itertools


DATASET_SETTINGS = {"Adult": {
        "vocab_path": "./data/uci_adult/vocabulary.json",
        "columns": ["age", "workclass", "fnlwgt", "education", "education-num",
                    "marital-status", "occupation", "relationship", "race",
                    "sex", "capital-gain", "capital-loss", "hours-per-week",
                    "native-country", "income"],
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['black','female'],
        "target_variable": "income",
        "target_value": " >50K"},
    "LSAC":{},
    "COMPAS":{
        'vocab_path':"./data/compas/vocabulary.json", 
        'columns':["juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
                    "age", "c_charge_degree", "c_charge_desc", "age_cat", "sex", "race",
                    "is_recid"],
        "sensitive_column_names": ['race','sex'],
        "sensitive_column_values": ['black','female'],
        "target_variable": "is_recid",
        "target_value": " Yes"
        }
    }


class Dataset(Dataset):

    def __init__(self, dataset_name, path, hide_sensitive_columns=True):
        '''
        path - str, path to the data csv file
        columns - list, names of all columns in the file
        target_variable - str, column which is the target
        target_value - ?, value which is considered as success (1)
        vocab_path - str, path to the vocab file which stores all possible values for each categorical column
        sensitive_column_names - list, names of protected attributes
        sensitive_column_values - list, values of attributes which are to be protected
        hide_sensitive_columns - bool, whether to hide (delete) sensitive columns
        '''
        super().__init__()

        vocab_path = DATASET_SETTINGS[dataset_name]["vocab_path"]
        columns = DATASET_SETTINGS[dataset_name]["columns"]
        sensitive_column_names = DATASET_SETTINGS[dataset_name]["sensitive_column_names"]
        sensitive_column_values = DATASET_SETTINGS[dataset_name]["sensitive_column_values"]
        target_variable = DATASET_SETTINGS[dataset_name]["target_variable"]
        target_value = DATASET_SETTINGS[dataset_name]["target_value"]

        self.hide_sensitive_columns = hide_sensitive_columns

        # load data
        self.features = pd.read_csv(path, ',', names=columns)
        
        # create labels
        self.labels = self.features[target_variable].to_numpy()
        self.labels[self.labels == target_value] = 1
        self.labels[self.labels != target_value] = 0

        # turn protected group memberships into a single index
        # first create lists of all the values the sensitive columns can take:
        uniques = [tuple(self.features[col].unique()) for col in sensitive_column_names]
        # create a list of tuples of such values. This corresponds to a list of all protected groups
        self.index2values = itertools.product(*uniques)
        # create the inverse dictionary:
        self.values2index = {vals: index for index, vals in enumerate(self.index2values)}


        # remove target variable from features
        columns.remove(target_variable)
        self.sensitives = self.features[sensitive_column_names]
        if hide_sensitive_columns: # remove sensitive columns
            for c in sensitive_column_names:
                columns.remove(c)
        self.features = self.features[columns]

        ## convert categorical data into onehot
        # load vocab
        with open(vocab_path) as json_file:
            vocab = json.load(json_file)
        
        # we already mapped target var values to 0 and 1 before
        del vocab[target_variable]
        

        # fill nan values in COMPAS dataset
        if dataset_name == 'COMPAS':
            self.features['c_charge_desc'].fillna('nan', inplace=True)

        for c in columns:
            if c in vocab:
                vals = list(vocab[c])
                val2int = {vals[i]:i for i in range(len(vals))} # map possible value to integer
                self.features[c] = self.features[c].apply(lambda x: nn.functional.one_hot(torch.Tensor([val2int[x]]).long(), num_classes=len(vals)).flatten())
            else: # feature is a scalar
                self.features[c] = self.features[c].apply(lambda x: torch.Tensor([x]))

        

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = list(self.features.iloc[index].to_numpy())
        x = torch.cat(x, dim=0)

        y = self.labels[index]

        s = tuple(self.sensitives.iloc[index])
        s = self.values2index[s]

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

    adult_dataset = Dataset("Adult", path="./data/uci_adult/train.csv")
    print('Example 1 of Adult set: \n',adult_dataset[1])

    compas_dataset = Dataset('COMPAS', path="./data/compas/train.csv")
    print('Example 1 of COMPAS set: \n',compas_dataset[1])

    #TODO: Add LSAC dataset
