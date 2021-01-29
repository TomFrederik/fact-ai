# Fairness without Demographics through Adversarially Reweighted Learning
This folder contains a slightly modified version of the implementation provided by
the authors of "Fairness without Demographics through Adversarially Reweighted Learning".

The original can be found at
https://github.com/google-research/google-research/tree/master/group_agnostic_fairness
It is licensed under the Apache 2.0 license, a copy of which can be found under `LICENSE`.

## Changes compared to the original
There are four groups of changes we made:

**Date preprocessing:** We were unable to run the IPython
notebooks provided by the authors, and even after making small changes to run them,
they produced incorrect CSV files that couldn't be read by the Python code.
We therefore wrote our own script for data preparation, based on these notebooks
(see `prepare_data.py` in the parent folder). We then slightly adapted the Python
code to read in the CSV files created by our script correctly.

**Renaming:** For convenience, we changed some of the model and dataset
names to match those we used in our own implementation.

**File output:** We added some command line options to specify the location
for the output files.

**Encoding:** The default in the code uses an embedding dimension of 32,
we used one-hot encoding instead (as in our own implementation).

**Seeding:** The original code doesn't use random seeds, we added support for
that (the seed can be set via a command line argument).

## Installation

Run the following:
```bash
virtualenv -p python3 .
source ./bin/activate

pip3 install -r group_agnostic_fairness/requirements.txt
```

## Data preparation
This code uses the same data as our own implementation (the `/data/` folder is symlinked into
this directory). So no additional preparation is needed.

## Training and Inference

Training and evaluation for a single model and dataset can be run using
```bash
python -m group_agnostic_fairness.main_trainer
```
Refer to the test cases in <model_name>_model_test.py files to understand the workflow.

To recreate the average results over multiple runs that we use, run
```bash
cd group_agnostic_fairness
for dataset in LSAC COMPAS Adult; do
    for model in baseline ARL IPW; do
        ./multiple_runs.sh "$dataset" "$model"
    done
done
```
(check our notebook at `/results.ipynb` to see how these results can be used)
