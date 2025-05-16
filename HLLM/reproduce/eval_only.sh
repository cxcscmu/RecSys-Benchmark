#!/bin/bash
#SBATCH --job-name=hllm_ml1m_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --output=hllm_ml1m_eval_%j.out
#SBATCH --error=hllm_ml1m_eval_%j.err

# Activate your virtual environment
source ~/hllm_venv/bin/activate

# Navigate to the HLLM code directory
cd /home/voberoi/HLLM/code

CKPT_DIR="../checkpoints/ml1m_run/HLLM-0.pth"
OUTPUT_FILE="../checkpoints/ml1m_run/pytorch_model.bin"
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Converting ZeRO stage 3 checkpoint to single file..."
    python $CKPT_DIR/zero_to_fp32.py $CKPT_DIR/checkpoint $OUTPUT_FILE
    if [ $? -ne 0 ]; then
        echo "Error converting checkpoint"
        exit 1
    fi
    echo "Checkpoint converted successfully"
fi

# Run the evaluation script
python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  --dataset ml1m \
  --train_batch_size 16 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir ../checkpoints/ml1m_run \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir ~/scratch/llms/tinyllama \
  --user_pretrain_dir ~/scratch/llms/tinyllama \
  --text_path /home/voberoi/HLLM/information \
  --gradient_checkpointing True \
  --stage 3 \
  --model_file ../checkpoints/ml1m_run/pytorch_model.bin \
  --val_only True



