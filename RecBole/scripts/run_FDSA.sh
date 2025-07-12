#!/bin/bash
#SBATCH --job-name=fdsa_ml1m
#SBATCH --partition=general           
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --output=fdsa_ml1m_%j.out        
#SBATCH --error=fdsa_ml1m_%j.err         

source /data/user_data/voberoi/GRU4REC/recysy_env/bin/activate  
cd /data/user_data/voberoi/GRU4REC

python run_FSDA.py
