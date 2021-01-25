import numpy as np
import math
import os
import json
import itertools

def get_path(base_path, model, dataset, lr, bs, num_steps):
    path = os.path.join(base_path, f'{dataset}_{model}_{lr}_{bs}_{num_steps}')
    return path

def subtract(dict1, dict2):
    if isinstance(dict1, dict):
        assert dict1.keys() == dict2.keys()
        return {k: subtract(dict1[k], dict2[k]) for k in dict1}
    else:
        return dict1 - dict2

def add(dict1, dict2):
    if isinstance(dict1, dict):
        assert dict1.keys() == dict2.keys()
        return {k: add(dict1[k], dict2[k]) for k in dict1}
    else:
        return dict1 + dict2

def valmap(dictionary, f):
    if isinstance(dictionary, dict):
        return {k: valmap(v, f) for k, v in dictionary.items()}
    else:
        return f(dictionary)

def square(dictionary):
    return valmap(dictionary, lambda x: x**2)

def div(dict1, dict2):
    if isinstance(dict1, dict):
        assert dict1.keys() == dict2.keys()
        return {k: div(dict1[k], dict2[k]) for k in dict1}
    else:
        return dict1 / dict2

models = ['baseline', 'IPW', 'ARL']
datasets = ['Adult', 'LSAC', 'COMPAS']

# load results
our_means = {}
our_stds = {}
their_means = {}
their_stds = {}

for model, dataset in itertools.product(models, datasets):
    path = get_path('./training_logs', model, dataset, 0.1, 128, 1000)
    with open(os.path.join(path, 'average.txt')) as f:
        our_means[(model, dataset)] = json.load(f)
    with open(os.path.join(path, 'std.txt')) as f:
        our_stds[(model, dataset)] = json.load(f)

    path = get_path('../google_research/group_agnostic_fairness/results', model, dataset, 0.1, 128, 1000)
    with open(os.path.join(path, 'average.txt')) as f:
        their_means[(model, dataset)] = json.load(f)
    with open(os.path.join(path, 'std.txt')) as f:
        their_stds[(model, dataset)] = json.load(f)

absolute_errors = subtract(our_means, their_means)
total_stds = valmap(add(square(our_stds), square(their_stds)), math.sqrt)
relative_errors = div(absolute_errors, total_stds)

key2index = {'min_auc':2, 'macro_avg_auc':1, 'micro_avg_auc':0, 'minority_auc':3}
index2key = [0]*len(key2index.keys())
for key in key2index:
    index2key[key2index[key]] = key

def create_line(model, dataset, result_entry):
    string = f'{dataset} & {model}'
    for i in range(len(index2key)):
        key = index2key[i]
        val = result_entry[key]
        if abs(val) > 2:
            string += r' & \textbf{' + f'{val:1.4f}' + r'}'
        else:
            string += f' & {val:1.4f}'
    string += '\\\\\n'
    return string

table = ''
for dataset in datasets:
    for i, model in enumerate(models):
        result_entry = relative_errors[(model, dataset)]
        new_line = create_line(model, dataset, result_entry)
        table += new_line
    table += r'\hline' + '\n'

print(table)
