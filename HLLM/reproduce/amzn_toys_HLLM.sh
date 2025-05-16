#!/bin/bash
#SBATCH --job-name=hllm_amzn_toys_train_2h100
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:H100:2       
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=flame-earlybirds
#SBATCH --qos=earlybird_qos
#SBATCH --time=48:00:00
#SBATCH --output=hllm_amzn_toys_train_%j.out
#SBATCH --error=hllm_amzn_toys_train_%j.err
#SBATCH --export=ALL

# Activate your virtual environment
source ~/hllm_venv/bin/activate

# Navigate to the HLLM code directory
cd /home/voberoi/HLLM/code
export NCCL_P2P_DISABLE=1

# Run the main training script
python3 main.py \
  --config_file overall/LLM_deepspeed2.yaml HLLM/HLLM.yaml \
  --loss nce \
  --epochs 5 \
  --train_batch_size 32 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir /data/user_data/voberoi/checkpoints/amzn-toys \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir ~/scratch/llms/tinyllama \
  --user_pretrain_dir ~/scratch/llms/tinyllama \
  --gradient_checkpointing True \
  --stage 3

