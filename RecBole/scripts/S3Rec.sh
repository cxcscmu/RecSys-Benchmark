#!/bin/bash

#SBATCH --job-name=s3rec_-amazon-sports_train
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=128G

#SBATCH --gres=gpu:A6000:8

#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate recsys_ben

export NCCL_P2P_DISABLE=1


source_dir="/data/user_data/chunings/RecSys-Benchmark/RecBole"

cd $source_dir

python3 run_recbole.py --model "S3Rec" --dataset "amzn-sports" --exp_name "s3rec_amzn-sports" --nproc 8 --config_files "configs/models/S3Rec.yaml configs/datasets/amzn.yaml configs/eval.yaml"
