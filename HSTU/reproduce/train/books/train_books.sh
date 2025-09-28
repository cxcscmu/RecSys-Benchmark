#!/bin/bash

#SBATCH --job-name="hstu_amzn-books_train"
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=96G

#SBATCH --gres=gpu:A6000:4

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="karrym@andrew.cmu.edu"


# enter a config env
eval "$(conda shell.bash hook)"
conda activate hstu

model="hstu"
dataset="amzn-books"

exp_name="${model}_${dataset}_valid"

nproc=1

source_dir="/home/karrym/capstone/RecSys-Benchmark/HSTU"
gin_config_file="configs/${dataset}/hstu-sampled-softmax-n512-large-final_train.gin"

cd $source_dir

RANDOM_PORT=$((10000 + RANDOM % 20000))
python3 main.py \
    --gin_config_file=$gin_config_file \
    --master_port=$RANDOM_PORT