import torch
from datasets import TabularDataset, CustomSubset

import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import time
import json
import os

def idx_mapping(x, target_grp, index2value):
    '''
    Binarizes group membership according to the target group and the index2value mapping
    
    Args:
        x: Tensor containing group memberships with values between 0 and 3
        target_grp: One of ['race', 'sex'].
        index2value: Mapping from values 0-3 to a tuple in race x sex.
    
    Returns:
        out: Tensor, binarized group memberships.
    
    '''
    out = torch.zeros_like(x)
    if target_grp == 'race':
        for key in index2value:
            if index2value[key][0] == 'Black':
                out[x == key] = 1
        return out
    elif target_grp == 'sex':
        for key in index2value:
            if index2value[key][1] == 'Female':
                out[x == key] = 1
        return out
    else:
        raise ValueError(f'Unexpected value for target_grp: {target_grp}')

def mem2label(target_grp):
    if target_grp == 'race':
        return {0:'White', 1:'Black'}
    if target_grp == 'sex':
        return {0:'Male', 1:'Female'}

def main(args):

    # set version
    version = int(time())

    # get mem2label dict
    mem2label_dict = mem2label(args.target_grp)

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create dataset
    train_dataset = TabularDataset(args.dataset, suffix=args.suffix)
    test_dataset = TabularDataset(args.dataset, test=True, suffix=args.suffix)
    print(train_dataset.index2values)
    print(test_dataset.index2values)
    
    # mapping memberships to 0/1
    print('Binarizing memberships...')
    train_memberships = idx_mapping(train_dataset.memberships, args.target_grp, train_dataset.index2values)
    test_memberships = idx_mapping(test_dataset.memberships, args.target_grp, test_dataset.index2values)
    print(train_memberships)
    print(test_memberships)
    
    # compute percentage of 0/1 group in train and test
    train_mean = torch.mean(train_memberships.float())
    test_mean = torch.mean(test_memberships.float())
    train_std = torch.std(train_memberships.float())
    test_std = torch.std(test_memberships.float())

    print(f'Percentage of {mem2label_dict[1]} in train: {train_mean} +- {train_std}')
    print(f'Percentage of {mem2label_dict[1]} in test: {test_mean} +- {test_std}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['Adult', 'COMPAS', 'LSAC'])
    parser.add_argument('--target_grp', choices=['race', 'sex'], required=True, help='Whether to predict race or sex of a person')
    parser.add_argument('--suffix', default='', help='Dataset suffix to specify other datasets than the defaults')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    
    args = parser.parse_args()

    main(args)