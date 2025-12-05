# HSTU

### Setup
Please ensure you have ```conda``` installed and create a new environment.
```bash
conda create -n hstu python=3.8
conda activate hstu
```
Please set up the environment using `requirements.txt` in `HSTU/` folder.
```bash
pip install -r requirements.txt
```

### Preprocesing dataset

We provide a bash script file `data_preprocess.sh` in `HSTU/reproduce/dataprocess/` to prepare data in necessary formats. Please run it before running training and testing scripts. 
Please update ```HSTU_DIR``` and ```DATA_SPLITS_DIR``` to your own HSTU directory and dataset splits directory:
```bash
# For example
HSTU_DIR="/home/karrym/capstone/RecSys-Benchmark/HSTU"
DATA_SPLITS_DIR="/data/group_data/cx_group/REC/data/benchmark_splits_corrected"
```
```DATA_SPLITS_DIR``` should be in this structure:
```bash
DATA_SPLITS_DIR/
├── amzn-beauty
│ ├── item
│ ├── train.inter
│ ├── valid.inter
│ └── test.inter
├── amzn-books
│ ├── item
│ ├── train.inter
│ ├── valid.inter
│ └── test.inter
├── amzn-sports
│ ├── item
│ ├── train.inter
│ ├── valid.inter
│ └── test.inter
├── amzn-toys
│ ├── item
│ ├── train.inter
│ ├── valid.inter
│ └── test.inter
├── ml-1m
│ ├── item
│ ├── user
│ ├── train.inter
│ ├── valid.inter
│ └── test.inter
```
When you complete the setup, run the following command inside `HSTU/reproduce/`:
```bash
./dataprocess/data_preprocess.sh
```
This script preprocesses and saves all datasets, so you only need to run it once before the experiments.

### Training and testing

We provide bash script files in `HSTU/reproduce/train/` and `HSTU/reproduce/test/` to train and evaluate HSTU model on each dataset. For example, to train and evaluate HSTU on ML-1M datasets, run:
```bash
./train/ml/train_ml.sh
```
and then
```bash
./test/ml/test_ml.sh
```
