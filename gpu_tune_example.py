import ray.tune as tune
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import ray
import time
import os


def process_all_batches(config, model, batches):
    for batch in batches:
        out = model(batch)
    tune.report(dummy_metric=0)

def create_model(device):
    model = nn.Sequential(nn.Linear(1000,1000),nn.Linear(1000,1000), nn.Linear(1000,1))
    model.to(device)
    return model

def create_data(device):
    return [torch.ones((1000,1000), device=device) for _ in range(100)] 

model = nn.Sequential(nn.Linear(1000,1000),nn.Linear(1000,1000), nn.Linear(1000,1))
model.to('cuda:0')
batches = [torch.ones((1000,1000), device='cuda:0') for _ in range(100)]

# set up models and data:
cpu_model = create_model('cpu')
cpu_data = create_data('cpu')

gpu_model = create_model('cuda:0')
gpu_data = create_data('cuda:0')

# set some parameters
config = {'dummy':tune.grid_search([1,2,3])}
path = f'./grid_search_gpu_example/'
num_cpus = 4
num_gpus = 1

# run gpu:
gpu_start = time.time()
analysis = tune.run(
    tune.with_parameters(
        process_all_batches,
        model=gpu_model,
        batches=gpu_data),
    resources_per_trial={
        'cpu': num_cpus,
        'gpu': num_gpus,
    },
    config=config,
    metric='dummy_metric',
    mode='max',
    local_dir=os.getcwd(),
    name=path 
    ) 
print(f'cuda run took {time.time() - gpu_start} seconds')


# run cpu:
cpu_start = time.time()
analysis = tune.run(
    tune.with_parameters(
        process_all_batches,
        model=cpu_model,
        batches=cpu_data),
    resources_per_trial={
        'cpu': num_cpus,
        'gpu': num_gpus,
    },
    config=config,
    metric='dummy_metric',
    mode='max',
    local_dir=os.getcwd(),
    name=path 
    ) 
print(f'cpu run took {time.time() - cpu_start} seconds')

