#!/bin/bash

#SBATCH --job-name="hstu_amzn-toys_test"
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=48G

#SBATCH --gres=gpu:1

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="bolinw@andrew.cmu.edu"


# enter a config env
eval "$(conda shell.bash hook)"
conda activate hstu

model="hstu"
dataset="amzn-toys"

exp_name="${model}_${dataset}_test"

nproc=1

source_dir="/data/user_data/bolinw/HSTU"
gin_config_file="configs/${dataset}/hstu-sampled-softmax-n512-large-final_test.gin"

cd $source_dir

python3 main.py \
    --gin_config_file=$gin_config_file \
    --master_port=12345