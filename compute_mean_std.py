'''
This script loads the results from multiple seeds for a certain setting and 
calculates the mean and std for all recorded metrics.
'''

import json
import os
import numpy
import argparse


def main(args):

    path = '' 

    for seed in range(args.num_seeds):
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('')

    args = parser.parse_args()

    main(args)