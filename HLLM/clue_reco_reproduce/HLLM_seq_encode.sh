#!/bin/bash
#SBATCH --job-name=HLLM-books-seq_encode
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=general       
#SBATCH --mem=512G 
#SBATCH --gres=gpu:A6000:1

#SBATCH --time=48:00:00

#SBATCH --mail-type=END


echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm

echo "activated"

user_pretrain_dir="TinyLlama-1.1B-Chat-v0.4"
item_pretrain_dir="TinyLlama-1.1B-Chat-v0.4"

checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/HLLM-amzn-books"

inter_path="/data/user_data/jingyuah/HLLM_weights/data/dataset"
info_path="/data/user_data/jingyuah/HLLM_weights/data/information"

epoch=5

num_cpus=4

embedding_output_dir="/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/HLLM_amzn-books"
# full merged item emb path 
item_embed_path="${embedding_output_dir}/item_embed_full_streamed.bin"
# seq data input path 
seq_data_path="/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/ordered_id_splits/test_input.tsv"

# seq_features.pkl: the seq features before inputting to user llm 
feature_output_path="${embedding_output_dir}/seq_features.pkl"
# seq_embd.bin: the user llm output sequence embedding 
seq_embed_output_path="${embedding_output_dir}/seq_embed.bin"

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))


# Item and User LLM are initialized by specific pretrain_dir.
python3 HLLM/code/seq_encode.py \
    --config_file HLLM/code/overall/LLM_deepspeed.yaml HLLM/code/HLLM/HLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --batch_size 160 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --best_model_path $item_pretrain_dir \
    --output_path $embedding_output_dir/clueweb-b-en.${shard}-of-${num_shards}.pkl \
    --item_embed_path $item_embed_path \
    --feature_output_path $feature_output_path \
    --seq_data_path $seq_data_path \
    --num_workers $num_cpus \
    --compute_seq_item_feature 



python3 HLLM/code/seq_encode.py \
    --config_file HLLM/code/overall/LLM_deepspeed.yaml HLLM/code/HLLM/HLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --batch_size 128 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --best_model_path $item_pretrain_dir \
    --output_path $embedding_output_dir/clueweb-b-en.${shard}-of-${num_shards}.pkl \
    --feature_output_path $feature_output_path \
    --num_workers $num_cpus \
    --seq_embed_output_path $seq_embed_output_path \
    --seq_encoding 
