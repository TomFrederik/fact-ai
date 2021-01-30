# Reproducibility Project: Fairness without Demographics through Adversarial Reweighted Learning

## Goal:
This repository re-implements `Fairness without Demographics through Adversarial Reweighted Learning` in PyTorch. The goal was to reproduce the results from the paper and to extend ARL to image data.

## Notebook
 To receive a guided end-to-end tour through our code, open `results.ipynb`. It contains the final results
 that we used for our report, you can also rerun parts of it or the entire notebook, and you can play around
 with many hyperparameters. The easiest way to run the notebook is to open it in Google Colab:
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomFrederik/fact-ai/blob/main/results.ipynb)
 *Please note the instructions at the top of the notebook (you need to uncomment a cell if you're using Colab)*

 If you want to run the notebook locally, you need to install the right Python environment first (see below). The notebook can reproduce
 all of our results except for the grid search to find the optimal hyperparameters. If you want to run the grid search yourself,
 see below or the end of the notebook for instructions.

## Organisation of this repo
`/data`  
The datasets used in the experiments:  
  - Adult  
  - LSAC  
  - COMPAS  
  - EMNIST_35
  - EMNIST_10
  
The tabular data is usable out of the box (though you can recreate it from scratch, see below). The image data (EMNIST) needs to be downloaded and preprocessed first (see below) if you want to use it.
  
`/paper_results`  
	Contains the results that were achieved by the authors of the original ARL paper in json format.  
  
`/job_scripts`  
	SLURM job scripts used in creating the results. Can be ignored.  
  
`/grid_search`  
	raw outputs, checkpoints and logs of our grid search  
  
`/training_logs`  
	raw outputs, checkpoints and logs of scripts
  
`./`  
	The root folder contains the code necessary to prepare the data, run all experiments and analyse the results. For a guided tour we recommend checking out `results.ipynb`.  
  
  
## Installing the environment
Execute the following commands to install the required packages and activate the environment.
Note that for installing on macOS, you need to remove the package cudatoolkit from the environment 
file or select a different available version.
```bash
conda env create -f environment.yml
conda activate fact-ai
```
  
## Creating the dataset
Simply run `prepare_data.sh` in the project root directory to download
and preprocess the datasets. Alternatively you can download the datasets
yourself and then run `python prepare_data.py` (URLs and filepaths
can be found in the shell script).


## Finding optimal hyperparameters
To execute grid searches for all models and datasets with default settings run the following command:
```bash
python get_opt_hparams.py --num_workers 2
```
(you can of course adjust `--num_workers` and may also want to set `--num_cpus`.
The optimal hyperparameters will be saved to `optimal_hparams.json`. More details can be found
at the end of the notebook.
### WARNING: This command can take multiple hours, depending on your machine. You can also use the already supplied optimal parameters.
