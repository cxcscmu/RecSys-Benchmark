#!/bin/bash
#SBATCH --job-name=taste_encode
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general   
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate taste


# Clueweb id map path
id_map_path="/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/cwid_to_id.tsv"
# ClueWeb-Reco
seq_data_path="/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/ordered_id_splits/test_input.tsv"


# TASTE amzn-books ckpt 
dataset="amzn-books"
data_dir="/data/group_data/cx_group/REC/data/TASTE"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/TASTE_${dataset}"
checkpoint_number="checkpoint-72500" 
best_model_path=${checkpoint_dir}/${checkpoint_number} 

# set 00 only 
output_dir=/data/group_data/cx_group/REC/ClueWeb-Reco/TASTE_exps/clueweb22b_TASTE-${checkpoint_number}
mkdir -p $output_dir


item_batch_size=4096
seq_batch_size=1024

# Seq: single shard 
python TASTE/encode.py \
    --seq_encoding \
    --id_map_path $id_map_path \
    --seq_data_path $seq_data_path \
    --best_model_path $best_model_path \
    --output_path $output_dir/clueweb22-seq-test.cweb.${shard}.pkl \
    --batch_size $seq_batch_size \
    --dataset_number_of_shards 1 \
    --dataset_shard_index 0 \
    --seed 2022 \
    --item_size 32  \
    --seq_size 256  \
    --num_passage 2  \
    --split_num 243  

