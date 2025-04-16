#!/bin/bash

#SBATCH --job-name=process_data_TASTE 
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



# enter a config env
eval "$(conda shell.bash hook)"
conda activate recsys_ben


# Configs
model="SASRec"
dataset_type="ml"
dataset="ml-1m"

# rewrite taste here, but leave model=SASRec just that Recbole detect a support model 
exp_name="/data/group_data/cx_group/REC/checkpoints/TASTE_${dataset}"

nproc=2

source_dir="/home/jingyuah/RecSys-Benchmark/RecBole"

model_config="${source_dir}/configs/models/${model}.yaml"
data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"

cd $source_dir

python run_recbole.py  \
    --dataset $dataset \
    --exp_name $exp_name \
    --nproc $nproc \
    --config_files "${model_config} ${data_config} ${eval_config}" \
    --data_preprocess
