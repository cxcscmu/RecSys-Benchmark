#!/bin/bash

dataset="ml-1m"
# the official split you exported
source_dir="benchmark_splits/${dataset}"
target_dir="dataset/${dataset}"

mkdir -p $target_dir

python code/transform_data_format.py \
    --source_dir $source_dir \
    --target_dir $target_dir
