#!/bin/bash
#SBATCH --job-name=HLLM-books
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --partition=general             
#SBATCH --mem=256G 
#SBATCH --gres=gpu:A6000:1

#SBATCH --time=48:00:00

#SBATCH --mail-type=END

echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm

echo "activated"

user_pretrain_dir="TinyLlama-1.1B-Chat-v0.4"
item_pretrain_dir="TinyLlama-1.1B-Chat-v0.4"

checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/HLLM_amzn-books"


inter_path="/data/user_data/jingyuah/HLLM_weights/data/dataset"
info_path="/data/user_data/jingyuah/HLLM_weights/data/information"

epoch=5

num_shards=64
shard=$1
item_batch_size=160
num_cpus=8
save_step=200

echo $shard 

# Clueweb id map path
id_map_path="/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/cwid_to_id.tsv"
embedding_output_dir="/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/HLLM_amzn-books/raw"


export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# use title, categories as item feature fields 


# Item and User LLM are initialized by specific pretrain_dir.
python3 HLLM/code/seq_encode.py \
    --config_file HLLM/code/overall/LLM_deepspeed.yaml HLLM/code/HLLM/HLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --item_size 160 \
    --seq_size 128 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --text_keys '[\"title\",\"categories\"]'  \
    --best_model_path $item_pretrain_dir \
    --id_map_path $id_map_path \
    --output_path $embedding_output_dir/clueweb-b-en.${shard}-of-${num_shards}.pkl \
    --batch_size $item_batch_size \
    --dataset_number_of_shards $num_shards \
    --dataset_shard_index $shard \
    --save_step $save_step \
    --num_workers $num_cpus \
    --item_encoding 


echo "Job Ends"
