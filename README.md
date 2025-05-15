<h1 align="center"> 
    ORBIT - Open Recommendation Benchmark for Reproducible Research with Hidden Tests
</h1>


<h4 align="center">
    <p>
        <a href="https://www.open-reco-bench.ai/leaderboard">ORBIT Benchmark</a> |
        <a href="https://huggingface.co/datasets/cx-cmu/ClueWeb-Reco">ClueWeb-Reco Dataset</a> |
        <a href="https://www.open-reco-bench.ai/leaderboard">Leaderboard</a> 
    <p>
</h4>



## Table of Contents

- [Setup](#setup)
- [Public Benchmark](#public-benchmark)
    - [Public Datasets Preparation](#public-datasets-preparation)
    - [Recbole-Supported Experiments](#recbole-supported-experiments)
    - [TASTE](#taste)
        - [TASTE Data Processing](#taste-data-processing)
        - [TASTE Experiments](#taste-experiments)
- [ClueWeb-Reco Benchmark](#clueweb-reco-benchmark)
    - [Dataset](#clueweb-reco-dataset)
    - [Submission and Evaluation](#clueweb-reco-benchmark-submission-and-evaluation)



## ClueWeb-Reco Benchmark Submission and Evaluation 


<!-- -------------------------- -->
<!-- -------------------------- -->
## Setup 


Install RecBole as per RecBole README: 

    cd RecSys-Benchmark/RecBole
    pip install -e . --verbose 



<!-- -------------------------- -->
<!-- -------------------------- -->
## Public Benchmark


<!-- -------------------------- -->
### Public Datasets Preparation 
We format public datasets into interactions, user-data, item-data. 
We use [RecSysDatasets - Conversion Tools](https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools) over this pre-processing step. 

While the item data processing part of the conversion tool do not work for AmazonReview 2023 datasets at the time we perform our experiements, we include a script that align the downloaded raw item data with the interactions exported by this conversion tool: `TASTE/reproduce/dataprocess/gen_item_column.sh`.   


<!-- -------------------------- -->
### Recbole-Supported Experiments 


We use [Recbole](https://github.com/RUCAIBox/RecBole) implementation of the following models. The configurations files we use over different models and datasets can be found under folder `Recbole/configs`. For more information and usage of custom configurations, please refer to the official [Recbole repository](https://github.com/RUCAIBox/RecBole).  

The scripts we use to launch experiments for each model can be found under folder `RecBole/scripts`. 
You can modify the `model` and `dataset` variables within these scripts to reproduce our experiments on the model and dataset desired. 

For example: 

    cd RecSys-Benchmark/RecBole
    sbatch scripts/run_SASRec.sh

*Note*: Remember to update the dataset you want to process in the scripts and associated paths.  


### TASTE 

We follow the [TASTE official opensource](https://github.com/OpenMatch/TASTE) to perform TASTE experiments. Please follow their official opensource for environment setup. 

#### TASTE Data processing 

We adapt the TASTE official data processing pipeline by running the following scripts. These scripts are adapted from the [TASTE official opensource](https://github.com/OpenMatch/TASTE). 

Firstly, format the dataset and its splits from the raw data described in [Public Datasets Preparation](#public-datasets-preparation). 

    sbatch RecBole/scripts/process_TASTE.sh

Secondly, generate item features: 

    sbatch TASTE/reproduce/dataprocess/gen_all_items.sh

Thirdly, generate training and evalution features: 

    sbatch TASTE/reproduce/dataprocess/build_train.sh

*Note*: Remember to update the dataset you want to process in the scripts and associated paths.  


#### TASTE Experiments 

The training scripts of TASTE experiments are in `TASTE/reproduce/train`. 
The testing scripts of TASTE experiments are in `TASTE/reproduce/test`. We use the lowest evaluation loss checkpoint to quickly perform testing.  

For example, to train and test TASTE on `ml-1m` dataset: 

    cd TASTE 
    sbatch reproduce/train/ml/train_ml.sh
    sbatch reproduce/test/ml/test_ml.sh


*Note*: Remember to update associated paths in these scripts.  



<!-- -------------------------- -->
<!-- -------------------------- -->
## ClueWeb-Reco Benchmark


<!-- -------------------------- -->
### ClueWeb-Reco Dataset

You can find [ClueWeb-Reco on Huggingface](https://huggingface.co/datasets/cx-cmu/ClueWeb-Reco). 

We provide two formats for the dataset: 

- ***pure interaction format***: input in columms [`session_id`, `cw_internal_id`, `timestamp`] that described all interaction in the dataset. 
- ***ordered cw id list format***: input in columns [`session_id`, `ordered_history_cw_internal_id`] that group all historically interacted items for each session. 

The ClueWeb-Reco dataset is structured as the following: 

    ## Source Files 
    -- cwid_to_id.tsv: mapping bewteen official ClueWeb22 docids and our internal docids

    ## Splits in pure interaction format: 
    -- interaction_splits: 
        -- valid_inter_input.tsv: input for validation dataset 
        -- valid_inter_target.tsv: validation dataset ground truth
        -- test_inter_input.tsv: input for testing dataset (ground truth hidden)


    ## Splits in ordered cw id list format: 
    -- ordered_id_splits: 
        -- valid_input.tsv: input for validation dataset 
        -- valid_target.tsv: validation dataset ground truth
        -- test_input.tsv: input for testing dataset (ground truth hidden) 

    ## Utility files for ClueWebApi usage and example processing on the ordered cw id list format  
    -- cw_data_processing: 
        - ClueWeb22Api.py: API to retrieve ClueWeb document information from official ClueWeb22 docids
        - example_dataset.py: example to load input data sequences with ClueWeb22Api



<!-- -------------------------- -->
### ClueWeb-Reco Benchmark Submission and Evaluation 

Your submitted prediction should be a binary file of the following format. Please make sure the submitted prediction binary files contain the ClueWeb internal IDs (0-indexing integer) instead of the official ClueWeb ids. 


    < 4 bytes int representing num_sessions><4 bytes int representing K><num_queries * K * sizeof(int) representing predicted clueweb internal ids>


We follow Recbole's evaluation to evaluate ClueWeb-Reco results, as in  `ClueWeb-Reco/get_metrics.sh` and `ClueWeb-Reco/get_metrics.py`. 
