import torch
import pytorch_lightning as pl
import baseline_model

import argparse 

# dict to access optimizers by name, if we need to use different opts.
OPT_BY_NAME = {'Adagrad': torch.optim.Adagrad}


def main(args):
    
    if args.model == 'ARL':
        raise NotImplementedError
    
    elif args.model == 'DRO':
        raise NotImplementedError
    
    elif args.model == 'baseline':
        raise NotImplementedError
    

    if args.dataset == 'Adult':
        raise NotImplementedError
    
    elif args.dataset == 'LSAC':
        raise NotImplementedError
    
    elif args.dataset == 'COMPAS':
        raise NotImplementedError






if __name__ == "__main__":
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    parser.add_argument(name='model',choices=['baseline', 'ARL', 'DRO'], required=True)
    parser.add_argument(name='dataset',choices=['Adult', 'LSAC', 'COMPAS'], required=True)
    parser.add_argument(name='prim_lr', default=0.1, type=float, help='Learning rate for primary network')
    parser.add_argument(name='adv_lr', default=0.1, type=float, help='Learning rate for adversarial network')
    parser.add_argument(name='batch_size', default=256, type=int)
    parser.add_argument(name='opt', choices=['Adagrad'], help='Name of optimizer')
    parser.add_argument(name='prim_hidden', default=[64,32], help='Number of hidden units in primary network')
    parser.add_argument(name='adv_hidden', default=[32], help='Number of hidden units in adversarial network')

    args = parser.parse_args()

    # run main loop
    main(args)
