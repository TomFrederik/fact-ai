import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import json


class dataset(Dataset):

    def __init__(self, path, columns, target_variable, target_value, vocab_path, sensitive_column_names, sensitive_column_values, hide_sensitive_columns):
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

        # load data
        self.features = pd.read_csv(path, ',', names=columns)
        
        # create labels
        self.labels = self.features[target_variable].to_numpy()
        self.labels[self.labels == target_value] = 1
        self.labels[self.labels != target_value] = 0

        # remove target variable from features
        columns.remove(target_variable)
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

        for c in vocab:
            vals = list(vocab[c])
            val2int = {vals[i]:i for i in range(len(vals))} # map possible value to integer
            self.features[c] = self.features[c].apply(lambda x: nn.functional.one_hot(torch.Tensor([val2int[x]]).long(), num_classes=len(vals)))

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x = self.features.iloc[index].to_numpy()
        y = self.labels[index]
        return x, y


if __name__ == '__main__':

    path = './data/uci_adult/train.csv'
    vocab_path = './data/uci_adult/vocabulary.json'

    columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    target_variable = "income"
    target_value = " >50K"

    dataset = dataset(path, columns, target_variable, target_value, vocab_path)

    print(dataset[1])