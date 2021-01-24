import torch
from datasets import CustomDataset, CustomSubset

import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import time
import json
import os

from MulticoreTSNE import MulticoreTSNE as TSNE

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
    os.makedirs(f'./tSNE_results/{args.dataset}/{args.target_grp}', exist_ok=True)

    # get mem2label dict
    mem2label_dict = mem2label(args.target_grp)

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create dataset
    train_dataset = CustomDataset(args.dataset)
    test_dataset = CustomDataset(args.dataset, test=True)

    # mapping memberships to 0/1
    print('Binarizing memberships...')
    train_memberships = idx_mapping(train_dataset.memberships, args.target_grp)
    test_memberships = idx_mapping(test_dataset.memberships, args.target_grp)

    
    # clustering train
    print('Clustering train set...')
    X_train = train_dataset.features.detach().numpy()
    Y_train = TSNE(n_jobs=2).fit_transform(X_train)
    np.save(f'./tSNE_results/{args.dataset}/{args.target_grp}/train_{version}.npy', Y_train)

    # plotting train
    print('Plotting train clusters...')
    plt.figure()
    plt.scatter(Y_train[train_memberships == 0,0], Y_train[train_memberships == 0,1], c='r', alpha=.6, label=mem2label_dict[0])
    plt.scatter(Y_train[train_memberships == 1,0], Y_train[train_memberships == 1,1], c='b', alpha=.6, label=mem2label_dict[1])
    plt.legend()
    plt.savefig(f'./tSNE_results/{args.dataset}/{args.target_grp}/train_{version}.pdf')

    # clustering test
    print('Clustering test set...')
    X_test = test_dataset.features.detach().numpy()
    Y_test = TSNE(n_jobs=2).fit_transform(X_test)
    np.save(f'./tSNE_results/{args.dataset}/{args.target_grp}/test_{version}.npy', Y_test)

    # plotting test
    print('Plotting test clusters...')
    plt.figure()
    plt.scatter(Y_test[test_memberships == 0,0], Y_test[test_memberships == 0,1], c='r', alpha=.6, label=mem2label_dict[0])
    plt.scatter(Y_test[test_memberships == 1,0], Y_test[test_memberships == 1,1], c='b', alpha=.6, label=mem2label_dict[1])
    plt.legend()
    plt.savefig(f'./tSNE_results/{args.dataset}/{args.target_grp}/test_{version}.pdf')


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['Adult', 'COMPAS', 'LSAC'])
    parser.add_argument('--target_grp', choices=['race', 'sex'], required=True, help='Whether to predict race or sex of a person')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    
    args = parser.parse_args()

    main(args)