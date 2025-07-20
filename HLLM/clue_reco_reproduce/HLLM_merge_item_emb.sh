#!/bin/bash
#SBATCH --job-name=HLLM-books-merge
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --partition=general             
#SBATCH --mem=1024G 
#SBATCH --gres=gpu:0

#SBATCH --time=48:00:00

#SBATCH --mail-type=END



# after all embeddings are computed, merge into binary 
# PLEASE UPDATE: clever_merge.py file for your custom path and shard numberbs 
python HLLM/code/clever_merge.py 