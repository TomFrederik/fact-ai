import math
import numpy as np
import os
import json
import itertools
from argparse import Namespace
import main

MAIN_RESULTS_HEADER = """
|Dataset|Method|Micro-avg AUC|Macro-avg AUC|Min AUC|Minority AUC|Accuracy|
|---|---|---|---|---|---|---|
"""

DEVIATION_HEADER = """
|Dataset|Method|Micro-avg AUC|Macro-avg AUC|Min AUC|Minority AUC|
|---|---|---|---|---|---|
"""

IDENTIFIABILITY_HEADER = """
|Target|Adult|LSAC|COMPAS|
|---|---|---|---|
"""

MAIN_RESULTS_KEYS = ['micro_avg_auc', 'macro_avg_auc', 'min_auc', 'minority_auc', 'accuracy']

DEVIATION_KEYS = ['micro_avg_auc', 'macro_avg_auc', 'min_auc', 'minority_auc']


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

def create_latex_line_scalar(row_key, result_entry, keys, bold_mask):
    if isinstance(row_key, tuple):
        string = ''
        for i, item in enumerate(row_key):
            string += f'{item}'
            if i + 1 < len(row_key):
                string += ' & '
    else:
        string = f'{row_key}'
    for key in keys:
        val = result_entry[key]
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

def create_markdown_line_scalar(row_key, result_entry, keys, bold_mask):
    if isinstance(row_key, tuple):
        string = ''
        for item in row_key:
            string += f'| {item}'
    else:
        string = f'| {row_key}'
    for key in keys:
        val = result_entry[key]
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

def run_models(seed, args, optimal_hparams, dataset_model_list):
    result_dict = {}
    for dataset, model in dataset_model_list:
        # don't overwrite the defaults:
        temp_args = Namespace(**vars(args))
        temp_args.log_dir = 'complete_run_logs'
        if model == 'IPW(S)':
            temp_args.model = 'IPW'
            temp_args.sensitive_label = False
        elif model == 'IPW(S+Y)':
            temp_args.model = 'IPW'
            temp_args.sensitive_label = True
        else:
            temp_args.model = model
        temp_args.dataset = dataset
        temp_args.seed = seed
        temp_args.dataset_type = 'image' if temp_args.dataset == 'EMNIST' else 'tabular'
        # set the optimal hyperparameters:
        for k, v in optimal_hparams[dataset][model].items():
            setattr(temp_args, k, v)

        # train and evaluate the model:
        result_dict[(dataset, model)] = main.main(temp_args)
    return result_dict

def result_list_to_dict(results, dataset_model_list, metrics):
    return {
        k: {
            metric: {
                'mean': np.mean([result_dict[k][metric] for result_dict in results]),
                'std': np.std([result_dict[k][metric] for result_dict in results])
            } for metric in metrics
        } for k in dataset_model_list
    }
