#!/bin/bash
#SBATCH --job-name=stamp_ml1m
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 

# Configs
model="STAMP"

dataset_type="ml" # amzn, ml
dataset="ml-1m" # amzn-beauty, amzn-books, amzn-toys, amzn-sports

exp_name="${model}_${dataset}"

model_config="configs/models/${model}.yaml"
data_config="configs/datasets/${dataset_type}.yaml"
eval_config="configs/eval.yaml"

python3 run_recbole.py \
    --model $model \
    --dataset $dataset \
    --exp_name $exp_name \
    --config_files "${model_config} ${data_config} ${eval_config}" --nproc 1 --port 2002
