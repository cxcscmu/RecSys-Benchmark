#!/bin/bash

#SBATCH --job-name=gen_all_items
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
python /home/jingyuah/REC_source/TASTE/gen_all_items.py  \
    --data_name Amazon  \
    --item_file /data/group_data/cx_group/REC/data/TASTE/${dataset}/item.txt  \
    --output item_name.jsonl  \
    --output_dir /data/group_data/cx_group/REC/data/TASTE/${dataset}  \
    --tokenizer google-t5/t5-base