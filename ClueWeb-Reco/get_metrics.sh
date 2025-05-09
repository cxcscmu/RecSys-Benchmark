#!/bin/bash
#SBATCH --job-name=pred_eval 
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general   
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate recsys_ben


# get results
predicted_file=""
target_path="/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/ordered_id_splits/valid_target.tsv"
python ClueWeb-Reco/get_metrics.py \
    --valid \
    --retrieval_result_path $predicted_file \
    --target_path $target_path \
    --Ks "[1, 10, 50, 100]"
