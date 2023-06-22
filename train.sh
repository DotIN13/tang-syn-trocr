#! /bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python -u train.py > log
