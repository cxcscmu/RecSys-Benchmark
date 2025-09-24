eval "$(conda shell.bash hook)"
conda activate hstu

# TODO: Change this to your own HSTU directory and dataset splits directory
HSTU_DIR="/home/karrym/capstone/RecSys-Benchmark/HSTU"
DATA_SPLITS_DIR="/data/group_data/cx_group/REC/data/benchmark_splits"

cd $HSTU_DIR
mkdir -p tmp

# Copy dataset splits exported from RecBole，skip if already exists
if [ ! -e tmp/$(ls $DATA_SPLITS_DIR/ | head -n1) ]; then
  cp -r $DATA_SPLITS_DIR/* tmp/
else
  echo "Files already exist in tmp/, skipping copy."
fi

python3 $HSTU_DIR/preprocess_public_data.py