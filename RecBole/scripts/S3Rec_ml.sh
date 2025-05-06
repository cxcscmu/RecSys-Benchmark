#!/bin/bash

#SBATCH --job-name=s3rec_load_with_8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 
#SBATCH --exclude=babel-13-13,babel-13-29,babel-13-1,babel-6-29,babel-5-31
#SBATCH --exclude=babel-3-[17,21,25],babel-4-[13,29,33],babel-7-13,babel-0-[19,37]

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=128G

#SBATCH --gres=gpu:8

#SBATCH --time=2:00:00

# enter a config env
eval "$(conda shell.bash hook)"
conda activate recsys_ben

export NCCL_P2P_DISABLE=1

# Configs

source_dir="/data/user_data/chunings/RecSys-Benchmark/RecBole"

cd $source_dir

python3 run_recbole.py --model "S3Rec" --dataset "ml-1m" --exp_name "s3rec_ml-1m" --nproc 8 --config_files "configs/models/S3Rec.yaml configs/datasets/ml.yaml configs/eval.yaml"
