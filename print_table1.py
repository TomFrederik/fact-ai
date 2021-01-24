import numpy as np
import os
import json
import itertools

def get_path(base_path, model, dataset, seed_run_version=0):
    path = os.path.join(base_path, dataset, model, f'seed_run_version_{seed_run_version}', 'mean_std.json')
    return path

models = ['baseline', 'DRO', 'ARL']
datasets = ['Adult', 'LSAC', 'COMPAS']

# load results
results = {}

for (model, dataset) in itertools.product(models, datasets):
    path = get_path('./training_logs', model, dataset)
    with open(path) as f:
        new_dict = json.load(f)
    results[f'{model}_{dataset}'] = new_dict

key2index = {'min_auc':2, 'macro_avg_auc':1, 'micro_avg_auc':0, 'minority_auc':3, 'accuracy':4}
index2key = [0]*len(key2index.keys())
for key in key2index:
    index2key[key2index[key]] = key



# create line
def create_line(model, dataset, result_entry, max_idcs):
    string = f'{dataset} & {model}'
    for i in range(len(index2key)):
        key = index2key[i]
        mean = result_entry[key]['mean']
        std = result_entry[key]['std']
        if max_idcs[i] == 1: # print max values in bold font
            string += r' & \textbf{'+f'{mean:1.4f}' + r'} $\pm$ \textbf{' + f'{std:1.4f}' + r'}'
        else:
            string += f' & {mean:1.4f}' + r' $\pm$ ' + f'{std:1.4f}'
    
    string += '\\\\\n'
    return string


def get_max_per_dataset(dataset):
    """computes indices of maximum values over all methods for a given dataset"
    Args:
        dataset: String specifying the dataset
    Returns:
        idcs: Binary Numpy array of shape (num_models, num_metrics) indicating which method achieves the best result for the given dataset on a metric
    """
    idcs = np.zeros((len(models), len(index2key)))
    means = np.zeros((len(models), len(index2key)))

    # load results in to np array
    for i, model in enumerate(models):
        model_results = results[f'{model}_{dataset}']
        for j in range(len(index2key)):
            key = index2key[j]
            means[i,j] = model_results[key]['mean']
    
    # compute argmax
    idcs[np.argmax(means, axis=0), np.arange(len(index2key))] = 1
    return idcs


models = ['baseline', 'DRO', 'ARL']
datasets = ['Adult', 'LSAC', 'COMPAS']

# load results
results = {}

for (model, dataset) in itertools.product(models, datasets):
    path = get_path('./training_logs', model, dataset)
    with open(path) as f:
        new_dict = json.load(f)
    results[f'{model}_{dataset}'] = new_dict

key2index = {'micro_avg_auc':0, 'macro_avg_auc':1, 'min_auc':2, 'minority_auc':3, 'accuracy':4}
index2key = [0]*len(key2index)
for key in key2index:
    index2key[key2index[key]] = key


# create table
table = ''
for dataset in datasets:
    max_idcs = get_max_per_dataset(dataset)
    for i, model in enumerate(models):
        result_entry = results[f'{model}_{dataset}']
        new_line = create_line(model, dataset, result_entry, max_idcs[i])
        table += new_line


# save table
with open('table_1.txt','w') as f:
    f.write(table)

