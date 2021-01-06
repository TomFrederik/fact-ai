import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import json


DATASET_SETTINGS = {"Adult": {
    "path": "./data/uci_adult/train.csv",
    "vocab_path": "./data/uci_adult/vocabulary.json",
    "columns": ["age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race",
                "sex", "capital-gain", "capital-loss", "hours-per-week",
                "native-country", "income"],
    "sensitive_column_names": ['race','sex'],
    "sensitive_column_values": ['black','female'],
    "target_variable": "income",
    "target_value": " >50K"}}


class Dataset(Dataset):

    def __init__(self, dataset_name, hide_sensitive_columns=True):
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

        path = DATASET_SETTINGS[dataset_name]["path"]
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

        # remove target variable from features
        columns.remove(target_variable)
        if hide_sensitive_columns: # remove sensitive columns
            self.sensitives = self.features[sensitive_column_names]
            for c in sensitive_column_names:
                columns.remove(c)
        self.features = self.features[columns]

        ## convert categorical data into onehot
        # load vocab
        with open(vocab_path) as json_file:
            vocab = json.load(json_file)
        
        # we already mapped target var values to 0 and 1 before
        del vocab[target_variable]
        
        for c in columns:
            if c in vocab:
                vals = list(vocab[c])
                val2int = {vals[i]:i for i in range(len(vals))} # map possible value to integer
                self.features[c] = self.features[c].apply(lambda x: nn.functional.one_hot(torch.Tensor([val2int[x]]).long(), num_classes=len(vals)))

        
        # deleted - makes no sense to encode the hidden features
        '''
        if self.hide_sensitive_columns:
            for c in sensitive_column_names:
                if c in vocab:
                    vals = list(vocab[c])
                    val2int = {vals[i]:i for i in range(len(vals))} # map possible value to integer
                    self.sensitives[c] = self.sensitives[c].apply(lambda x: nn.functional.one_hot(torch.Tensor([val2int[x]]).long(), num_classes=len(vals)))
        '''

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x = self.features.iloc[index].to_numpy()
        y = self.labels[index]

        if not self.hide_sensitive_columns:
            s = None
        else:
            s = self.sensitives.iloc[index].to_numpy()

        return x, y, s


if __name__ == '__main__':

    dataset = Dataset("Adult")

    print(dataset[1])