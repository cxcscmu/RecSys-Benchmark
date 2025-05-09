#!/bin/bash

#SBATCH --job-name=DeepFM_ml-1m
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=preempt
#SBATCH --exclude=babel-3-[13,17,21,25],babel-4-[13,29,33],babel-7-13,babel-0-[19,37],babel-10-9,babel-13-[1,13,29],babel-6-29,babel-5-31

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=128G

#SBATCH --gres=gpu:1

#SBATCH --time=720:00:00


# disable NCCL P2P for some nodes, (from Jiongnan)
if [[ "$(hostname)" =~ ^(shire-2-(9|5)|babel-8-5|babel-4-(1|5|9|13|17|21|25|29)|babel-6-(5|9|13|29)|babel-7-(1|5|9)|babel-12-(5|9|13)|babel-13-(1|5|9|13|17|21|25|29)|babel-14-(1|5|9|13|17|21|25|29|37)|babel-5-15|babel-10-17|babel-0-19|babel-11-25|babel-9-3)$ ]]; then
  export NCCL_P2P_DISABLE=1
fi


# enter a config env
eval "$(conda shell.bash hook)"
conda activate recbole-py310


# Configs
model="DeepFM"

dataset_type="ml" # amzn, ml
dataset="ml-1m" # amzn-beauty, amzn-books, amzn-toys, amzn-sports 

exp_name="${model}_${dataset}"

# nproc=1

source_dir="/home/karrym/capstone/RecBole"

model_config="${source_dir}/configs/models/${model}.yaml"
data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"

cd $source_dir

python run_recbole.py  \
    --dataset $dataset \
    --exp_name $exp_name \
    --config_files "${model_config} ${data_config} ${eval_config}" \