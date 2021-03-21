import os
from matplotlib import pyplot as plt
import json

dataset = "LSAC"

path = os.path.join("final_logs", "complete_runs", dataset, "ARL", "seed_run_version_seclr", dataset+"_gridsearch_results.json")

"""
###################################################################################
# Save gridsearch results

data = {}
data["batchsize"] = []
data["prim_lr"] = []
data["sec_lr"] = []
data["auc"] = []

with open(path, "r") as f:
    line = f.readline()

    while line:
        data["batchsize"].append(int(line[53:56].strip()))
        data["prim_lr"].append(float(line[67:73].strip()))
        data["sec_lr"].append(float(line[78:84].strip()))
        data["auc"].append(float(line[114:123].strip()))

        line = f.readline()


with open('LSAC_gridsearch_results.json', 'w') as outfile:
    json.dump(data, outfile)
###################################################################################
"""

###################################################################################
# Read gridsearch results
with open(path, "r") as f:
    data = json.load(f)

    plt.figure(figsize=(12, 8))
    plt.hist(data["auc"], bins=80)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylabel("Number of hparam configurations", fontsize=16)
    plt.xlabel("AUC", fontsize=16)
    plt.show()
