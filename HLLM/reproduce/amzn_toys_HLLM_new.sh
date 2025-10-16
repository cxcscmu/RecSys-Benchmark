#!/bin/bash
#SBATCH --job-name=hllm_amzn_toys_train
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=preempt
#SBATCH --time=48:00:00
#SBATCH --output=outputs/hllm_amzn_toys_train_%j.out
#SBATCH --error=outputs/hllm_amzn_toys_train_%j.err

# Activate your virtual environment
eval "$(conda shell.bash hook)"
conda activate hllmenv

# Navigate to the HLLM code directory
cd /data/user_data/bolinw/RecSys-Benchmark/HLLM/code
export NCCL_P2P_DISABLE=1

dataset="amzn-toys"
exp_name="hllm_${dataset}"
checkpoint_dir="/data/user_data/bolinw/checkpoints/${dataset}"

# Run the main training script
python3 main.py \
  --config_file overall/LLM_deepspeed_amzn.yaml HLLM/HLLM.yaml \
  --loss nce \
  --epochs 5 \
  --train_batch_size 32 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir $checkpoint_dir \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir /data/user_data/bolinw/scratch/llms/tinyllama \
  --user_pretrain_dir /data/user_data/bolinw/scratch/llms/tinyllama \
  --gradient_checkpointing True \
  --stage 3 \
  --exp_name $exp_name \
  --dataset $dataset \

