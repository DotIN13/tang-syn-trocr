#! /bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py > log