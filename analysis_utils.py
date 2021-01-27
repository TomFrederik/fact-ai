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
        try:
            with open(path) as f:
                results[(dataset, model)] = json.load(f)
        except FileNotFoundError:
            print(f"Didn't find results for {dataset}, {model}, skipping")
    return results

def is_max(result_dict, metrics):
    # list of all datasets
    datasets = set(dataset for dataset, model in result_dict)
    # for each dataset, find the maximum over all models with that dataset
    max_vals = {
        dataset: {
            metric: max(
                result_dict[(ds, model)][metric]["mean"]
                for ds, model in result_dict
                if ds == dataset)
            for metric in metrics
        }
        for dataset in datasets
    }
    # now for each item, check whether it is the maximum
    return {
        (dataset, model): {
            metric: v[metric]["mean"] == max_vals[dataset][metric]
            for metric in metrics
        }
        for (dataset, model), v in result_dict.items()
    }

def create_latex_line(row_key, result_entry, keys, bold_mask):
    if isinstance(row_key, tuple):
        string = ''
        for i, item in enumerate(row_key):
            string += f'{item}'
            if i + 1 < len(row_key):
                string += ' & '
    else:
        string = f'{row_key}'
    for key in keys:
        val = result_entry[key]['mean']
        if bold_mask[key]:
            string += r' & \textbf{' + f'{val:1.3f}' + r'}'
        else:
            string += f' & {val:1.3f}'
    string += '\\\\\n'
    return string

def create_latex_line_with_std(row_key, result_entry, keys, bold_mask):
    if isinstance(row_key, tuple):
        string = ''
        for i, item in enumerate(row_key):
            string += f'{item}'
            if i + 1 < len(row_key):
                string += ' & '
    else:
        string = f'{row_key}'
    for key in keys:
        mean = result_entry[key]['mean']
        std = result_entry[key]['std']
        if bold_mask[key]:
            string += r' & \textbf{'+f'{mean:1.4f}' + r'} $\pm$ \textbf{' + f'{std:1.4f}' + r'}'
        else:
            string += f' & {mean:1.4f}' + r' $\pm$ ' + f'{std:1.4f}'
    string += '\\\\\n'
    return string

def create_markdown_line(row_key, result_entry, keys, bold_mask):
    if isinstance(row_key, tuple):
        string = ''
        for item in row_key:
            string += f'| {item}'
    else:
        string = f'| {row_key}'
    for key in keys:
        val = result_entry[key]['mean']
        if bold_mask[key]:
            string += f' | **{val:1.2f}**'
        else:
            string += f' | {val:1.2f}'
    string += ' |\n'
    return string

def create_markdown_line_with_std(row_key, result_entry, keys, bold_mask):
    if isinstance(row_key, tuple):
        string = ''
        for item in row_key:
            string += f'| {item}'
    else:
        string = f'| {row_key}'
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
    for row_key in result_dict:
        result_entry = result_dict[row_key]
        bold_mask = bold_dict[row_key]
        table += line_func(row_key, result_entry, keys, bold_mask)
    return table
