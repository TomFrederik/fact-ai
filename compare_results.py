import numpy as np
import math
import os
import json
import itertools

def get_path(base_path, model, dataset, seed_run_version=0):
    path = os.path.join(base_path, dataset, model, f'seed_run_version_{seed_run_version}', 'mean_std.json')
    return path

def get_their_path(base_path, model, dataset):
    path = os.path.join(base_path, f"{model}_{dataset}.json")
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

def load_result_dict(base_path, datasets, models, path_func):
    results = {}
    for dataset, model in itertools.product(datasets, models):
        path = path_func(base_path, model, dataset)
        with open(path) as f:
            results[(dataset, model)] = json.load(f)
    return results

def create_line(model, dataset, result_entry, keys, bold_mask):
    string = f'{dataset} & {model}'
    for key in keys:
        val = result_entry[key]
        if bold_mask[key]:
            string += r' & \textbf{' + f'{val:1.4f}' + r'}'
        else:
            string += f' & {val:1.4f}'
    string += '\\\\\n'
    return string

def create_table(result_dict, keys, bold_dict):
    table = ''
    for dataset, model in result_dict:
        result_entry = result_dict[(dataset, model)]
        bold_mask = bold_dict[(dataset, model)]
        table += create_line(model, dataset, result_entry, keys, bold_mask)
    return table


if __name__ == '__main__':
    models = ['baseline', 'DRO', 'ARL']
    datasets = ['Adult', 'LSAC', 'COMPAS']

    # load results
    our_results = load_result_dict('training_logs', datasets, models, get_path)
    for k in our_results:
        # we don't have accuracy values from the paper
        del our_results[k]["accuracy"]

    our_means = {k1: {k2: v2["mean"] for k2, v2 in v1.items()} for k1, v1 in our_results.items()}
    our_stds = {k1: {k2: v2["std"] for k2, v2 in v1.items()} for k1, v1 in our_results.items()}

    their_results = load_result_dict('paper_results', datasets, models, get_their_path)

    their_means = {k1: {k2: v2["mean"] for k2, v2 in v1.items()} for k1, v1 in their_results.items()}
    their_stds = {k1: {k2: v2["std"] for k2, v2 in v1.items()} for k1, v1 in their_results.items()}

    absolute_errors = subtract(our_means, their_means)
    total_stds = valmap(add(square(our_stds), square(their_stds)), math.sqrt)
    relative_errors = div(absolute_errors, total_stds)

    bold_dict = valmap(relative_errors, lambda x: abs(x) >= 2)

    keys = ['micro_avg_auc', 'macro_avg_auc', 'min_auc', 'minority_auc']
    table = create_table(relative_errors, keys, bold_dict)

    print(table)
