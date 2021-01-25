import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('pdf')

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import traceback
from functools import reduce
import argparse
import os



SETTINGS = {
        'models': ['baseline', 'ARL', 'DRO', 'IPW(S)', 'IPW(S+Y)'],
        'colors': ['green', 'red', 'blue', 'orange', 'yellow'],
        'min_auc': 'Minimum group AUC',
        'micro_avg_auc': 'Micro-average AUC',
        'macro_avg_auc': 'Macro-average AUC',
        'minority_auc': 'AUC of minority group'
    }


# taken from https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data



    
def main(args: argparse.Namespace):    
    
    for model, color in zip(SETTINGS['models'], SETTINGS['colors']):
        
        paths = []
        dfList = []
        
        for root, dirs, files in os.walk(f'{args.log_dir}/{args.dataset}/{model}'):
            for file in files:
                if 'tfevents' in file:
                    paths = paths + [os.path.join(root, file)]      
                    
        for i, path in enumerate(paths):
            df = tflog2pandas(path)    
            df = df[(df['metric'] == f'{args.split}/{args.metric}')]
            df.rename(columns={'value': f'value_{i}'}, inplace=True)      
            dfList = dfList + [df]
                
        try:
            df_all_seeds = reduce(lambda x, y: pd.merge(x, y, on=['step', 'metric'], how='outer'), dfList)
        except:
            print(f'Values for {model} not found.')
            continue
                       
        # compute mean and std for metrics over all seeds   
        df_all_seeds['avg'] = df_all_seeds.filter(like='value').mean(axis=1)
        df_all_seeds['std'] = df_all_seeds.filter(like='value').std(axis=1)
        
        steps = df_all_seeds['step'][df_all_seeds['metric'] == f'{args.split}/{args.metric}']
        avg = df_all_seeds['avg'][df_all_seeds['metric'] == f'{args.split}/{args.metric}']
        std = df_all_seeds['std'][df_all_seeds['metric'] == f'{args.split}/{args.metric}']
        
        plt.plot(steps, avg, color=color, label=f'{model}')
        plt.fill_between(steps, avg-std, avg+std, facecolor=color, alpha=0.1)
        
 
    plt.ylabel('%s' % SETTINGS[args.metric])
    plt.xlabel('Number of steps')  
    plt.legend()
    plt.savefig(f'{args.dataset}_{args.split}_{args.metric}.pdf')
    

if __name__ == '__main__':
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS', 'FairFace', 'FairFace_reduced', 'colorMNIST'], default='LSAC')
    parser.add_argument('--log_dir', default='final_logs', type=str)
    parser.add_argument('--split', choices=['training', 'validation', 'test'], default='validation')
    parser.add_argument('--metric', choices=['min_auc', 'macro_avg_auc', 'micro_avg_auc', 'minority_auc'], default='min_auc')    
    
    args: argparse.Namespace = parser.parse_args()
    
    # run main loop
    main(args)

    