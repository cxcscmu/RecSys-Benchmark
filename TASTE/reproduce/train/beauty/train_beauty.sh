#!/bin/bash

#SBATCH --job-name=TASTE_beauty
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=256G

#SBATCH --gres=gpu:8

#SBATCH --exclude=babel-1-31,babel-4-21

#SBATCH --time=48:00:00



# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste


model_path="google-t5/t5-base"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints"



dataset="amzn-beauty"
data_dir="/data/group_data/cx_group/REC/data/TASTE"

export WANDB_PROJECT="RecSys-Benchmark"
exp_name="TASTE_${dataset}"

lr=7e-4
bz=48
n_epochs=30
nproc=8


NCCL_P2P_DISABLE=1 torchrun --nproc-per-node=$nproc TASTE/train.py  \
    --output_dir "${checkpoint_dir}/${exp_name}"  \
    --model_name_or_path $model_path  \
    --do_train  \
    --save_steps 10  \
    --eval_steps 10  \
    --train_path "${data_dir}/${dataset}/train_name.jsonl"  \
    --eval_path "${data_dir}/${dataset}/valid_name.jsonl"  \
    --per_device_train_batch_size $bz \
    --per_device_eval_batch_size $bz \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate $lr  \
    --warmup_ratio 0.1 \
    --q_max_len 256  \
    --p_max_len 32  \
    --seed 2022  \
    --num_train_epochs $n_epochs  \
    --evaluation_strategy steps  \
    --save_total_limit 2 \
    --metric_for_best_model "eval_loss" \
    --logging_steps 1 \
    --report_to wandb \
    --run_name $exp_name