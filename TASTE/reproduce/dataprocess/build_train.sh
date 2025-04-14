#!/bin/bash

#SBATCH --job-name=build_train
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=256G

#SBATCH --gres=gpu:0

#SBATCH --time=48:00:00

# enter a config env
eval "$(conda shell.bash hook)"
conda activate taste


dataset="amzn-books"

python /home/jingyuah/REC_source/TASTE/build_train.py  \
    --data_name Amazon  \
    --train_file /data/group_data/cx_group/REC/data/TASTE/${dataset}/valid.txt  \
    --item_file /data/group_data/cx_group/REC/data/TASTE/${dataset}/item.txt  \
    --item_ids_file /data/group_data/cx_group/REC/data/TASTE/${dataset}/item_name.jsonl  \
    --output valid_name.jsonl  \
    --output_dir /data/group_data/cx_group/REC/data/TASTE/${dataset}/  \
    --seed 2022  \
    --tokenizer google-t5/t5-base

python /home/jingyuah/REC_source/TASTE/build_train.py  \
    --data_name Amazon  \
    --train_file /data/group_data/cx_group/REC/data/TASTE/${dataset}/train.txt  \
    --item_file /data/group_data/cx_group/REC/data/TASTE/${dataset}/item.txt  \
    --item_ids_file /data/group_data/cx_group/REC/data/TASTE/${dataset}/item_name.jsonl  \
    --output train_name.jsonl  \
    --output_dir /data/group_data/cx_group/REC/data/TASTE/${dataset}/  \
    --seed 2022  \
    --tokenizer google-t5/t5-base
 