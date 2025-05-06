# RecSys-Benchmark
Ressys benchmark code repo


## Setup 


Install RecBole as per RecBole README: 

    cd RecSys-Benchmark/RecBole
    pip install -e . --verbose 


## Data Preparation 
Data should be formatted into interactions, user-data, item-data. 
We use [RecSysDatasets - Conversion Tools](https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools) over this pre-processing step. 



## ClueWeb-Reco

The ClueWeb-Reco dataset is structured as the following: 


    ## Source Files 
    -- cwid_to_id.tsv: mapping bewteen official ClueWeb22 docids and our internal docids

    ## Splits in pure interaction format 
    -- interaction_splits: 
        -- valid_inter_input.tsv: input for validation dataset 
        -- valid_inter_target.tsv: validation dataset ground truth
        -- test_inter_input.tsv: input for testing dataset (ground truth hidden)

    ## Splits in ordered cw id list format
    -- ordered_id_splits: 
        -- valid_input.tsv: input for validation dataset 
        -- valid_target.tsv: validation dataset ground truth
        -- test_input.tsv: input for testing dataset (ground truth hidden) 

    ## Utility files for ClueWebApi usage and example processing on the ordered cw id list format  
    -- cw_data_processing: 
        - ClueWeb22Api.py: API to retrieve ClueWeb document information from official ClueWeb22 docids
        - example_dataset.py: example to load input data sequences with ClueWeb22Api


Your submitted prediction should be a binary file of the following format. Please make sure the submitted prediction binary files contain the ClueWeb internal IDs (0-indexing integer) instead of the official ClueWeb ids. 


    < 4 bytes int representing num_sessions><4 bytes int representing K><num_queries * K * sizeof(int) representing predicted clueweb internal ids>


We follow Recbole's evaluation, as in  `ClueWeb-Reco/get_metrics.sh` and `ClueWeb-Reco/get_metrics.py`. 