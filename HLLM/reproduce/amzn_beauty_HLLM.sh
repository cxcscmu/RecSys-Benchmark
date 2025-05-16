#!/bin/bash
#SBATCH --job-name=hllm_amzn_beauty_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --output=hllm_amzn_beauty_train_%j.out
#SBATCH --error=hllm_amzn_beauty_train_%j.err
#SBATCH --exclude=babel-15-32,babel-9-11,babel-4-33

# Activate your virtual environment
source ~/hllm_venv/bin/activate

# Navigate to the HLLM code directory
cd /home/voberoi/HLLM/code
export NCCL_P2P_DISABLE=1

# Run the main training script
python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  --epochs 5 \
  --train_batch_size 16 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir /data/user_data/voberoi/checkpoints/amzn-beauty \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir ~/scratch/llms/tinyllama \
  --user_pretrain_dir ~/scratch/llms/tinyllama \
  --gradient_checkpointing True \
  --stage 3

