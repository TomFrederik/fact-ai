#!/bin/bash

set -eu

mkdir -p results
mkdir -p logs
cd ..

lr=0.1
bs=128
dataset="$1"
model="$2"
steps=100

echo "Dataset and model:"
echo "$dataset $model"

run () {
    echo "Starting run with seed $1"
    mkdir -p "group_agnostic_fairness/results/${model}_${dataset}"
    # the trainer outputs lots of deprecation and other warnings,
    # hide everything to not clutter up this scripts output
    mkdir -p "group_agnostic_fairness/logs/${model}_${dataset}"
    python &> "group_agnostic_fairness/logs/${model}_${dataset}/$1.txt" \
	   -m group_agnostic_fairness.main_trainer \
	   --train_steps "$steps" \
	   --test_steps 100 \
	   --primary_learning_rate "$lr" \
	   --adversary_learning_rate "$lr" \
	   --batch_size "$bs" \
	   --dataset "$dataset" \
	   --model_name "$model" \
	   --seed "$1" \
	   --output_file_path "group_agnostic_fairness/results/${model}_${dataset}/$1.json"
    echo "Finished run with seed $1"
}

# Parallelize across N processes
# See https://unix.stackexchange.com/a/436713
N=4

for seed in {1..10}; do
    run "$seed" &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait

cd -
echo "Averaging results"
python average_results.py "results/${model}_${dataset}"
echo "Done."
