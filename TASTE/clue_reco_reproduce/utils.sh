#!/bin/bash
#SBATCH --job-name=taste_pred
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general   
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

# eval "$(conda shell.bash hook)"
# conda activate taste

echo "Starts"

# merge and convert item, sequence embed 
python TASTE/clue_reco_reproduce/convert_embeddings.py

# similarity_pred: use DiskANN (https://github.com/microsoft/DiskANN) to compute embedding KNNS 
base_dir="/data/group_data/cx_group/REC/ClueWeb-Reco/TASTE_exps/clueweb22b_TASTE-checkpoint-72500"
embed_file="${base_dir}/item_embed.bin"
query_file="${base_dir}/test_input_embed.bin"
predict_file="${base_dir}/test_prediction.bin"
K=1000
./DiskANN/build/apps/utils/compute_groundtruth  --data_type float --dist_fn mips \
        --base_file $embed_file \
        --query_file $query_file \
        --gt_file $predict_file \
        --K $K

    
echo "Ends"