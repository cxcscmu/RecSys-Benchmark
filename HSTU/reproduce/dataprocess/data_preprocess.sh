/data/user_data/bolinw/HSTU/RecBole/scripts/write_ml-1m.sh
/data/user_data/bolinw/HSTU/RecBole/scripts/write_amzn.sh

eval "$(conda shell.bash hook)"
conda activate hstu

python3 /data/user_data/bolinw/HSTU/preprocess_public_data.py