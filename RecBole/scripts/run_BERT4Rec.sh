#!/bin/bash

#SBATCH --job-name=BERT4Rec_ml1m
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --mem=16G

#SBATCH --gres=gpu:2

#SBATCH --time=48:00:00


# enter a config env
eval "$(conda shell.bash hook)"
conda activate recsys_ben


# Configs
model="BERT4Rec"


dataset_type="ml" # amzn, ml
dataset="ml-1m" # amzn-beauty, amzn-books, amzn-toys, amzn-sports 

exp_name="${model}_${dataset}"

nproc=2

source_dir="RecSys-Benchmark/RecBole"

model_config="${source_dir}/configs/models/${model}.yaml"
data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"

cd $source_dir

python run_recbole.py  \
    --dataset $dataset \
    --exp_name $exp_name \
    --config_files "${model_config} ${data_config} ${eval_config}" \
