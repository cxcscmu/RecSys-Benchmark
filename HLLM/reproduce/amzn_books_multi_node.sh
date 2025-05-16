#!/bin/bash
#SBATCH --job-name=hllm_amzn_books_train_4n1g      
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=general                       
#SBATCH --time=48:00:00
#SBATCH --output=hllm_amzn_books_train_%j.out     
#SBATCH --error=hllm_amzn_books_train_%j.err       
#SBATCH --exclude=babel-15-32,babel-9-11,babel-4-33
#SBATCH --export=ALL

source ~/hllm_venv/bin/activate
cd   /home/voberoi/HLLM/code
export NCCL_P2P_DISABLE=1

export master_addr=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export master_port=$(( 29000 + RANDOM % 1000 ))
export nnodes=$SLURM_JOB_NUM_NODES             
export nproc_per_node=1                        
# ---------------------------------------

srun -n $SLURM_NTASKS --ntasks-per-node=1 --export=ALL \
     bash -c 'export node_rank=$SLURM_NODEID; \
              python3 -u main.py \
              --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
              --loss nce \
              --epochs 5 \
              --train_batch_size 16 \
              --MAX_TEXT_LENGTH 256 \
              --MAX_ITEM_LIST_LENGTH 10 \
              --checkpoint_dir /data/user_data/voberoi/checkpoints/amzn-books \
              --optim_args.learning_rate 1e-4 \
              --item_pretrain_dir ~/scratch/llms/tinyllama \
              --user_pretrain_dir ~/scratch/llms/tinyllama \
              --gradient_checkpointing True \
              --stage 3'

