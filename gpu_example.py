import torch
import torch.nn as nn
import ray
import time

@ray.remote(num_cpus=3, num_gpus=1)
def process_batch(network, batch):
    out = network(batch)
    print(torch.cuda.memory_allocated(0)/1024**3)
    return out

@ray.remote(num_cpus=3)
def process_batch_cpu(network, batch):
    out = network(batch)
    print(torch.cuda.memory_allocated(0)/1024**3)
    return out



model = nn.Sequential(nn.Linear(1000,1000),nn.Linear(1000,1000), nn.Linear(1000,1))
model.to('cuda:0')
batches = [torch.ones((1000,1000), device='cuda:0') for _ in range(100)]

cpu_model = nn.Sequential(nn.Linear(1000,1000),nn.Linear(1000,1000), nn.Linear(1000,1))
cpu_batches = [torch.ones((1000,1000)) for _ in range(100)]

ray.init(num_cpus=3, num_gpus=1)

gpu_start = time.time()
futures = [process_batch.remote(model, batch) for batch in batches]
results = ray.get(futures)
print(f'cuda run took {time.time() - gpu_start} seconds')

cpu_start = time.time()
futures = [process_batch_cpu.remote(cpu_model, batch) for batch in cpu_batches]
results = ray.get(futures)
print(f'cpu run took {time.time() - cpu_start} seconds')