#!/bin/bash

lr=0.1
bs=128
dataset="$1"
model="$2"
steps=1000

function run {
    echo "Starting run with seed $1"
    python &>/dev/null \
	   main.py \
	   --train_steps "$steps" \
	   --prim_lr "$lr" \
	   --batch_size "$bs" \
	   --dataset "$dataset" \
	   --model "$model" \
	   --seed "$1" \
	   --no_grid_search \
	   --tf_mode
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

echo "Averaging results"
python average_results.py "training_logs/${model}_${dataset}_${lr}_${bs}_${steps}"
echo "Done."
