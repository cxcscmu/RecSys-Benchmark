#!/bin/bash

#SBATCH --job-name=TASTE_toys
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=preempt  

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=256G

#SBATCH --gres=gpu:8

#SBATCH --time=200:00:00


# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste


model_path="/data/group_data/cx_group/self_rewarding_framework/jingyuah/self_reward_rs/pretrained_model/t5_base"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints"


dataset="amzn-toys"
data_dir="/data/group_data/cx_group/REC/data/TASTE"

export WANDB_PROJECT="RecSys-Benchmark"
exp_name="TASTE_${dataset}"

lr=7e-4
bz=48
n_epochs=30
nproc=8


# CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/jingyuah/REC_source/TASTE/train.py  \
NCCL_P2P_DISABLE=1 torchrun --nproc-per-node=$nproc /home/jingyuah/REC_source/TASTE/train.py  \
    --output_dir "${checkpoint_dir}/${exp_name}"  \
    --model_name_or_path $model_path  \
    --do_train  \
    --save_steps 5000  \
    --eval_steps 5000  \
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
    --logging_steps 100 \
    --report_to wandb \
    --run_name $exp_name