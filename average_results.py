import json
import argparse
import os
import statistics
import math

parser = argparse.ArgumentParser()

parser.add_argument('directory', help='The directory in which to average files')
args = parser.parse_args()

results = []

for entry in os.scandir(args.directory):
    if entry.name.endswith('.json'):
        print("Processing ", entry.path)
        with open(entry.path, 'r') as file:
            results.append(json.load(file))

average = {
    k: statistics.mean(x[k] for x in results)
    for k in results[0]
    if isinstance(results[0][k], float)
}
std = {
    k: statistics.stdev(x[k] for x in results)
    for k in results[0]
    if isinstance(results[0][k], float)
}
with open(os.path.join(args.directory, 'average.txt'), 'w') as file:
    json.dump(average, file, indent=4)
with open(os.path.join(args.directory, 'std.txt'), 'w') as file:
    json.dump(std, file, indent=4)

print("============Results============")
print("Accuracy:", average["accuracy"], "+-", std["accuracy"] / math.sqrt(len(results)))
print("Total AUC:", average["macro_avg_auc"], "+-", std["macro_avg_auc"] / math.sqrt(len(results)))
print("AUC subgroup 1 (minority AUC):", average["minority_auc"], "+-", std["minority_auc"] / math.sqrt(len(results)))
print("===============================")
