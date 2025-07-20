#!/bin/bash
#SBATCH --job-name=cos_prediction
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --partition=general    
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=1024G 
#SBATCH --time=48:00:00

#SBATCH --mail-type=END
echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm


base_dir="/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/HLLM_amzn-books"
embed_file="${base_dir}/item_embed_full_streamed.bin"
query_file="${base_dir}/seq_embed.bin"
predict_file="${base_dir}/prediction.bin"
K=1000

python HLLM/code/custom_eval.py \
        --seq_emb_path $query_file \
        --item_emb_path $embed_file \
        --k $K \
        --output_binary_path $predict_file

echo "Job Ends" 