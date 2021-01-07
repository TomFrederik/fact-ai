'''
This script simply removes the additional colon '.' from the income column in the test 
dataset of the UCI adult dataset.

'''



import pandas as pd



columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                    "marital-status", "occupation", "relationship", "race",
                    "sex", "capital-gain", "capital-loss", "hours-per-week",
                    "native-country", "income"]
                    

path = './data/uci_adult/test.csv'


with open(path) as f:
    # load data
    features = pd.read_csv(path, ',', names=columns, skiprows=1)
    features['income'] = features['income'].apply(lambda x: x[:-1])
    features.to_csv(path, header=False)