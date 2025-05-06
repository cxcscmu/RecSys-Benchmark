#!/bin/bash

#SBATCH --job-name=sasrecf_-amzn-books_train
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 
#SBATCH --exclude=babel-13-13,babel-13-29,babel-13-1,babel-6-29,babel-5-31
#SBATCH --exclude=babel-3-[17,21,25],babel-4-[13,29,33],babel-7-13,babel-0-[19,37]

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=128G

#SBATCH --gres=gpu:1

#SBATCH --time=10:00:00

eval "$(conda shell.bash hook)"
conda activate recsys_ben


source_dir="/data/user_data/chunings/RecSys-Benchmark/RecBole"

cd $source_dir

python3 run_recbole.py --model "SASRecF" --dataset "amzn-books" --exp_name "SASRecf_amzn-books" --nproc 1 --config_files "configs/models/sasrecf.yaml configs/datasets/amzn.yaml configs/eval.yaml"
