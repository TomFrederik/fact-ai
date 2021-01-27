# Reproducibility Project: Fairness without Demographics through Adversarial Reweighted Learning

## Goal:
This repository re-implements `Fairness without Demographics through Adversarial Reweighted Learning` in PyTorch. The goal was to reproduce the results from the paper and to extend ARL to image data.

## Organisation of this repo
`/data`  
The datasets used in the experiments:  
  - Adult  
  - LSAC  
  - COMPAS  
  - EMNIST  
  
The data should be usable out of the box. If it doesn't or the data is not directly available in the repo, create the datasets as described below.  
  
`/paper_results`  
	Contains the results that were achieved by the authors of the original ARl paper in json format.  
  
`/job_scripts`  
	SLURM job scripts used in creating the results. Can be ignored.  
  
`/grid_search`  
	raw outputs, checkpoints and logs of the scripts that run a grid search  
  
`/training_logs`  
	raw_outputs, checkpoints and logs of the scripts than run individual training runs
  
`/final logs`
	raw_outputs, checkpoints and logs of the runs that were used in to produce the final results
  
`./`  
	The root folder contains the code necessary to prepare the data, run all experiments and analyse the results. For a guided tour we recommend checking out `results.ipynb`.  
  
  
## Installing the environment
Execute the following commands to install the required packages and activate the environment  
`conda env create -f environment.yml`  
`conda activate fact-ai`
  
## Creating the dataset
Simply run `prepare_data.sh` in the project root directory to download
and preprocess the datasets. Alternatively you can download the datasets
yourself and then run `python prepare_data.py` (URLs and filepaths
can be found in the shell script).

## Notebook
 To receive a guided end-to-end tour through our code, open `results.ipynb` e.g. via Jupyter or Google Colab. There you can inspect the already available outputs and re-run experiments at will. The notebook contains all you need to reproduce the results, except the grid searches to find the optimal hyperparameters. To re-run the grid searches, follow the instructions below or in the notebook.

## Finding optimal hyperparameters
To execute grid searches for all models and datasets with default settings run the following command
`python get_opt_hparams.py --num_workers 2`
The optimal hyperparameters will be saved to `optimal_hparams.json`.
### WARNING: This command can take multiple hours, depending on your machine. You can also use the already supplied optimal parameters.





