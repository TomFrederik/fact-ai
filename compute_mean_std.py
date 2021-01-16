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
    path = f'./{args.log_dir}/{args.dataset}/{args.model}/seed_run_version_{args.seed_run_version}/seed_'

    all_scores = {}
    for seed in range(1, args.num_seeds+1):
        seed_path = os.path.join(path + str(seed), 'auc_scores.json')
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
    result_path =  f'./{args.log_dir}/{args.dataset}/{args.model}/seed_run_version_{args.seed_run_version}/mean_std.json'
    with open(result_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seeds', default=10, help='number of seeds to average')
    parser.add_argument('--log_dir', default='training_logs')
    parser.add_argument('--model', choices=['baseline', 'ARL', 'DRO', 'IPW'], required='True')
    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS'], required='True')
    parser.add_argument('--seed_run_version', default=0)


    args = parser.parse_args()

    main(args)