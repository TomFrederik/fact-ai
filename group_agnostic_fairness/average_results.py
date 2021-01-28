import json
import argparse
import os
import statistics
import numpy as np
import math

# convert from our key names to theirs
key_dict = {
    "accuracy": "accuracy",
    "micro_avg_auc": "auc",
    "minority_auc": "auc subgroup 1"
}

parser = argparse.ArgumentParser()

parser.add_argument('directory', help='The directory in which to average files')
args = parser.parse_args()

results = []

for entry in os.scandir(args.directory):
    if entry.name.endswith('.json'):
        print("Processing ", entry.path)
        with open(entry.path, 'r') as file:
            data = json.load(file)
            renamed_data = {k: data[key_dict[k]] for k in data}
            results.append(renamed_data)

mean_std_dict = {
    k: {
        "mean": np.mean([x[k] for x in results]),
        "std": np.std([x[k] for x in results])
    } for k in results[0]
}


with open(os.path.join(args.directory, 'mean_std.json'), 'w') as file:
    json.dump(mean_std_dict, file, indent=4)
