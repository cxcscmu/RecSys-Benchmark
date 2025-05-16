#!/bin/bash
#SBATCH --job-name=hllm_ml1m_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --output=hllm_ml1m_train_%j.out
#SBATCH --error=hllm_ml1m_train_%j.err

# Activate your virtual environment
source ~/hllm_venv/bin/activate

# Navigate to the HLLM code directory
cd /home/voberoi/HLLM/code

# Run the training script
python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  --dataset ml1m \
  --train_batch_size 16 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir ../checkpoints/ml1m_eval_run \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir ~/scratch/llms/tinyllama \
  --user_pretrain_dir ~/scratch/llms/tinyllama \
  --text_path /home/voberoi/HLLM/information \
  --gradient_checkpointing True \
  --stage 3 \
  --epochs 1 \
  --model_file ../checkpoints/ml1m_run/HLLM-0.pth 