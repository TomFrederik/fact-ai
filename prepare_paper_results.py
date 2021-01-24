import numpy as np
import statistics
import math
import shutil
from pathlib import Path
import os
import json
import itertools

def argmin(xs):
    """Returns argmin and min of an iterable."""
    return min(enumerate(xs), key=lambda x: x[1])

path = Path("paper_results")
# create a list so we don't iterate over the copies we'll create
files = list(path.iterdir())

for item in files:
    print(item.name)
    # Path.with_stem is only in Python 3.9+, so use with_name instead for compatibility
    shutil.copy(item, item.with_name(item.stem + "_raw.json"))
    with item.open() as f:
        raw = json.load(f)
    results = {}
    idx, min_mean = argmin(x["mean"] for x in raw["subgroups"])
    results["min_auc"] = {
        "mean": min_mean,
        "std": raw["subgroups"][idx]["std"]
    }
    results["macro_avg_auc"] = {
        "mean": statistics.mean(x["mean"] for x in raw["subgroups"]),
        # standard error of the mean
        "std": statistics.mean(x["std"] for x in raw["subgroups"]) / math.sqrt(len(raw["subgroups"])),
    }
    results["micro_avg_auc"] = raw["overall"]
    # by convention, last subgroup contains the minority
    results["minority_auc"] = raw["subgroups"][-1]

    with item.open('w') as f:
        json.dump(results, f)
