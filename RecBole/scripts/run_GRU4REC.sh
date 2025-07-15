#!/bin/bash
#SBATCH --job-name=gru4rec_ml1m
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=gru4rec_ml1m_%j.out
#SBATCH --error=gru4rec_ml1m_%j.err


# Configs
model="GRU4Rec"

dataset_type="ml" # amzn, ml
dataset="ml-1m" # amzn-beauty, amzn-books, amzn-toys, amzn-sports

exp_name="${model}_${dataset}"

source_dir="RecSys-Benchmark/RecBole"

model_config="${source_dir}/configs/models/${model}.yaml"
data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"


cd $source_dir

python3 run_recbole.py \
    --model $model \
    --dataset $dataset \
    --exp_name $exp_name \
    --config_files "${model_config} ${data_config} ${eval_config}" --nproc 1 --port 2002
