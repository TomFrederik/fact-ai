'''
This script loads the results from multiple seeds for a certain setting and 
calculates the mean and std for all recorded metrics.
'''

import json
import os
import numpy as np
import argparse


def main(args):

    # get results of all seeds
    if args.model == 'IPW':
        if args.sensitive_label:
            path = f'./{args.log_dir}/{args.dataset}/IPW(S+Y)/seed_run_version_{args.seed_run_version}'
        else:
            path = f'./{args.log_dir}/{args.dataset}/IPW(S)/seed_run_version_{args.seed_run_version}'
    else:
        path = f'./{args.log_dir}/{args.dataset}/{args.model}/seed_run_version_{args.seed_run_version}'

    print(f'Loading results from {path}.')

    all_scores = {}
    for seed in range(1, args.num_seeds+1):
        seed_path = os.path.join(path, f'seed_{seed}', 'auc_scores.json')
        with open(seed_path, 'r') as f:
            seed_scores = json.load(f)
            for key in seed_scores:
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(seed_scores[key])    

    # create result dict and save mean and std
    results = {}
    for key in all_scores:
        results[key] = {}
        results[key]['mean'] = np.mean(all_scores[key])
        results[key]['std'] = np.std(all_scores[key])
    
    # save dict to json
    result_path =  os.path.join(path, 'mean_std.json')
    print(f'Saving results to {result_path}')
    with open(result_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seeds', default=10, type=int, help='number of seeds to average')
    parser.add_argument('--log_dir', default='training_logs')
    parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO', 'IPW'], required='True')
    parser.add_argument('--sensitive_label', default=False, action='store_true', help='Whether to use the label Y in IPW')
    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS', 'EMNIST_35', 'EMNIST_10'], required='True')
    parser.add_argument('--seed_run_version', default=0)


    args = parser.parse_args()

    main(args)