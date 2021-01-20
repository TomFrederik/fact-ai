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
index2key = {key2index[k]:k for k in key2index}

# create line
def create_line(model, dataset, result_entry):
    string = f'{dataset} & {model}'
    for i in range(len(index2key)):
        key = index2key[i]
        mean = result_entry[key]['mean']
        std = result_entry[key]['std']
        string += f' & {mean:1.5f}' + r' $\pm$ ' + f'{std:1.5f}'
    
    string += '\\\\\n'
    return string


table = ''
for (dataset, model) in itertools.product(datasets, models):
    result_entry = results[f'{model}_{dataset}']
    new_line = create_line(model, dataset, result_entry)
    table += new_line

# save table
with open('table_1.txt','w') as f:
    f.write(table)

