import numpy as np
import math
import os
import json
import itertools

def get_path(base_path, model, dataset, seed_run_version=0):
    path = os.path.join(base_path, dataset, model, f'seed_run_version_{seed_run_version}', 'mean_std.json')
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

models = ['baseline', 'DRO', 'ARL']
datasets = ['Adult', 'LSAC', 'COMPAS']

# load results
our_results = {}

for model, dataset in itertools.product(models, datasets):
    path = get_path('./training_logs', model, dataset)
    with open(path) as f:
        our_results[(model, dataset)] = json.load(f)
        # we don't have accuracy values from the paper
        del our_results[(model, dataset)]["accuracy"]

our_means = {k1: {k2: v2["mean"] for k2, v2 in v1.items()} for k1, v1 in our_results.items()}
our_stds = {k1: {k2: v2["std"] for k2, v2 in v1.items()} for k1, v1 in our_results.items()}

their_results = {}

for model, dataset in itertools.product(models, datasets):
    path = os.path.join("paper_results", f"{model}_{dataset}.json")
    with open(path) as f:
        their_results[(model, dataset)] = json.load(f)

their_means = {k1: {k2: v2["mean"] for k2, v2 in v1.items()} for k1, v1 in their_results.items()}
their_stds = {k1: {k2: v2["std"] for k2, v2 in v1.items()} for k1, v1 in their_results.items()}

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

print(table)
