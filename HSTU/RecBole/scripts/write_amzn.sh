# enter a config env
eval "$(conda shell.bash hook)"
conda activate hstu


# Configs
model="DCN"
dataset_type="amzn"
source_dir="/data/user_data/bolinw/HSTU/RecBole"
model_config="${source_dir}/configs/models/${model}.yaml"
eval_config="${source_dir}/configs/eval.yaml"
nproc=1

# List of datasets to run
datasets=(
  "amzn-toys"
  "amzn-beauty"
  "amzn-sports"
  "amzn-books"
)

cd "$source_dir"

for dataset in "${datasets[@]}"; do
  exp_name="${model}_${dataset}"
  data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"

  echo "=== Running $model on $dataset (exp: $exp_name) ==="
  python run_recbole.py \
    --model="$model" \
    --dataset="$dataset" \
    --exp_name="$exp_name" \
    --nproc="$nproc" \
    --config_files="$model_config $data_config $eval_config" \
    --output_file="${source_dir}/log/${model}/${dataset}.log"
  echo "=== Finished $dataset ==="
  echo
done

echo "All experiments completed."