#!/bin/bash

#SBATCH --job-name=TASTE_books
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general     

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=300G

#SBATCH --gres=gpu:A6000:8
#SBATCH --exclude=babel-15-32,babel-9-11

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"


# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste


model_path="google-t5/t5-base"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints"


dataset="amzn-books"
data_dir="/data/group_data/cx_group/REC/data/TASTE"

export WANDB_PROJECT="RecSys-Benchmark"
exp_name="TASTE_${dataset}"

lr=4e-4
bz=48
n_epochs=30
nproc=8


NCCL_P2P_DISABLE=1 torchrun --nproc-per-node=$nproc TASTE/train.py  \
    --output_dir "${checkpoint_dir}/${exp_name}"  \
    --model_name_or_path $model_path  \
    --do_train  \
    --save_steps 2500  \
    --eval_steps 2500  \
    --train_path "${data_dir}/${dataset}/train_name.jsonl"  \
    --eval_path "${data_dir}/${dataset}/valid_name.jsonl"  \
    --per_device_train_batch_size $bz \
    --per_device_eval_batch_size $bz \
    --train_n_passages 10  \
    --num_passages 2  \
    --learning_rate $lr  \
    --ignore_data_skip \
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