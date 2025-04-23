#!/bin/bash

#SBATCH --job-name=SASRec_ml1m
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

eval "$(conda shell.bash hook)"
conda activate recsys_ben

model="NeuMF"
dataset_type="ml"
dataset="ml-1m"

exp_name="${model}_${dataset}"
nproc=2

source_dir="/home/jingyuah/RecSys-Benchmark/RecBole"

data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"
output_dir="benchmark_splits/${dataset}" 

python export_data_split.py \
    --model "$model" \
    --dataset_name "$dataset" \
    --output_dir "$output_dir" \
    --config_file_list "${data_config} ${eval_config}" \
