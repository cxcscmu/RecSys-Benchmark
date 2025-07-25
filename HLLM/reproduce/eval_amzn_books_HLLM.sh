#!/bin/bash
#SBATCH --job-name=hllm_books_eval_book_branch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --output=logs/hllm_books_eval_%j.out
#SBATCH --error=logs/hllm_books_eval_%j.err


eval "$(conda shell.bash hook)"
conda activate hllm

cd /home/jingyuah/RecSys-Benchmark/HLLM/code

user_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-Chat-v0.4"
item_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-Chat-v0.4"

dataset="amzn-books"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/HLLM_amzn-books"

info_path="/home/jingyuah/RecSys-Benchmark/HLLM/information"

# Item and User LLM are initialized by specific pretrain_dir.
python3 main.py \
    --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset $dataset \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --val_only True  



#   --model_file ../checkpoints/ml1m_run/pytorch_model.bin \



