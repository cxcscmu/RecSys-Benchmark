#!/bin/bash

#SBATCH --job-name=SASRec_ml1m
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

model="together"
nproc=2

python3 get_queries.py \
    --model "$model"
