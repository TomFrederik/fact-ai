import torch
from datasets import CustomDataset, CustomSubset

import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import time
import json
import os

def idx_mapping(x, target_grp):
    if target_grp == 'race':
        x[x == 1] = 0
        x[x > 1] = 1
        return x
    else:
        x[x == 2] = 0
        x[x > 0] = 1
        return x

def mem2label(target_grp):
    if target_grp == 'race':
        return {0:'White', 1:'Black'}
    if target_grp == 'sex':
        return {0:'Male', 1:'Female'}

def main(args):

    # set version
    version = int(time())

    # make sure dir exist
    #os.makedirs(f'./dataset_statistics/{args.dataset}/{args.target_grp}', exist_ok=True)

    # get mem2label dict
    mem2label_dict = mem2label(args.target_grp)

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create dataset
    train_dataset = CustomDataset(args.dataset, suffix=args.suffix)
    test_dataset = CustomDataset(args.dataset, test=True, suffix=args.suffix)

    # mapping memberships to 0/1
    print('Binarizing memberships...')
    train_memberships = idx_mapping(train_dataset.memberships, args.target_grp)
    test_memberships = idx_mapping(test_dataset.memberships, args.target_grp)

    # compute percentage of 0/1 group in train and test
    train_mean = torch.mean(train_memberships.float())
    test_mean = torch.mean(test_memberships.float())
    train_std = torch.std(train_memberships.float())
    test_std = torch.std(test_memberships.float())

    print(f'Percentage of {mem2label_dict[0]} in train: {train_mean} +- {train_std}')
    print(f'Percentage of {mem2label_dict[0]} in test: {test_mean} +- {test_std}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['Adult', 'COMPAS', 'LSAC'])
    parser.add_argument('--target_grp', choices=['race', 'sex'], required=True, help='Whether to predict race or sex of a person')
    parser.add_argument('--suffix', default='', help='Dataset suffix to specify other datasets than the defaults')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    
    args = parser.parse_args()

    main(args)