#!/bin/bash

#SBATCH --job-name=TASTE_ml
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=128G

#SBATCH --gres=gpu:1

#SBATCH --exclude=babel-1-31,babel-4-21

#SBATCH --time=48:00:00


# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste


dataset="ml-1m"
data_dir="/data/group_data/cx_group/REC/data/TASTE"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/TASTE_${dataset}"

CUDA_VISIBLE_DEVICES=0 python /home/jingyuah/REC_source/TASTE/inference.py  \
    --data_name ${dataset} \
    --data_dir ${data_dir} \
    --checkpoint_dir "runlog"  \
    --experiment_name ${dataset}  \
    --seed 2022  \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  \
    --eval_batch_size 512  \
    --Ks "[1, 10, 50, 100]" \
    --best_model_path ${checkpoint_dir}/checkpoint-25000
