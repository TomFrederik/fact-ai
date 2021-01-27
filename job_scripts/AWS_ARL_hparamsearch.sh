#!/bin/bash

python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 32 --prim_lr 0.001
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 64 --prim_lr 0.001
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 128 --prim_lr 0.001
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 256 --prim_lr 0.001
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 512 --prim_lr 0.001

python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 32 --prim_lr 0.01
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 64 --prim_lr 0.01
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 128 --prim_lr 0.01
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 256 --prim_lr 0.01
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 512 --prim_lr 0.01

python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 32 --prim_lr 0.1
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 64 --prim_lr 0.1
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 128 --prim_lr 0.1
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 256 --prim_lr 0.1
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 512 --prim_lr 0.1

python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 32 --prim_lr 1.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 64 --prim_lr 1.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 128 --prim_lr 1.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 256 --prim_lr 1.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 512 --prim_lr 1.0

python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 32 --prim_lr 2.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 64 --prim_lr 2.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 128 --prim_lr 2.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 256 --prim_lr 2.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 512 --prim_lr 2.0

python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 32 --prim_lr 5.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 64 --prim_lr 5.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 128 --prim_lr 5.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 256 --prim_lr 5.0
python main.py --model ARL --dataset EMNIST --num_workers 4 --no_grid_search --batch_size 512 --prim_lr 5.0
