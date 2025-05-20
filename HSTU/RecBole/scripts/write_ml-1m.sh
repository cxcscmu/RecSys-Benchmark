# enter a config env
eval "$(conda shell.bash hook)"
conda activate hstu


# Configs
model="DCN"
dataset_type="ml"
dataset="ml-1m"

exp_name="${model}_${dataset}"

nproc=1

source_dir="/data/user_data/bolinw/HSTU/RecBole"

model_config="${source_dir}/configs/models/${model}.yaml"
data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"

cd $source_dir

python run_recbole.py  \
    --model=$model \
    --dataset=$dataset \
    --exp_name=$exp_name \
    --nproc=$nproc \
    --config_files="${model_config} ${data_config} ${eval_config}" \
    --output_file="${source_dir}/log/${model}/${dataset}.log"