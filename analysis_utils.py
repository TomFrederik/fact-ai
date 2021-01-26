import math
import os
import json
import itertools

def get_our_path(base_path, model, dataset, seed_run_version=0):
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

def create_latex_line(model, dataset, result_entry, keys, bold_mask):
    string = f'{dataset} & {model}'
    for key in keys:
        val = result_entry[key]
        if bold_mask[key]:
            string += r' & \textbf{' + f'{val:1.2f}' + r'}'
        else:
            string += f' & {val:1.2f}'
    string += '\\\\\n'
    return string

def create_latex_line_with_std(model, dataset, result_entry, keys, bold_mask):
    string = f'{dataset} & {model}'
    for key in keys:
        mean = result_entry[key]['mean']
        std = result_entry[key]['std']
        if bold_mask[key]:
            string += r' & \textbf{'+f'{mean:1.4f}' + r'} $\pm$ \textbf{' + f'{std:1.4f}' + r'}'
        else:
            string += f' & {mean:1.4f}' + r' $\pm$ ' + f'{std:1.4f}'
    string += '\\\\\n'
    return string

def create_markdown_line(model, dataset, result_entry, keys, bold_mask):
    string = f'| {dataset} | {model}'
    for key in keys:
        val = result_entry[key]
        if bold_mask[key]:
            string += f' | **{val:1.2f}**'
        else:
            string += f' | {val:1.2f}'
    string += ' |\n'
    return string

def create_markdown_line_with_std(model, dataset, result_entry, keys, bold_mask):
    string = f'|{dataset} | {model}'
    for key in keys:
        mean = result_entry[key]['mean']
        std = result_entry[key]['std']
        if bold_mask[key]:
            string += f' | **{mean:1.4f} +- {std:1.4f}**'
        else:
            string += f' | {mean:1.4f} +- {std:1.4f}'
    string += '\n'
    return string

def create_table(result_dict, keys, bold_dict, line_func):
    table = ''
    for dataset, model in result_dict:
        result_entry = result_dict[(dataset, model)]
        bold_mask = bold_dict[(dataset, model)]
        table += line_func(model, dataset, result_entry, keys, bold_mask)
    return table
