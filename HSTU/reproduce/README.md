# HSTU

### Setup

Please set up the environment using `requirements.txt` in `HSTU/` folder.
```bash
pip install -r requirements.txt
```

### Preprocesing dataset

We provide a bash script file `data_preprocess.sh` in `HSTU/reproduce/dataprocess/` to prepare data in necessary formats. Please run it before running training and testing scripts. Inside `HSTU/reproduce/`, run:
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