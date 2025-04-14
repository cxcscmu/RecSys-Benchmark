# RecSys-Benchmark
Ressys benchmark code repo


## Setup 

<!-- Clone this project and all submodules: 

    git clone --recursive https://github.com/cxcscmu/RecSys-Benchmark.git

Update submodules if needed: 

    git submodule update --recursive --remote -->


Install RecBole as per RecBole README: 

    cd RecSys-Benchmark/RecBole
    pip install -e . --verbose 


## Data Preparation 
Data should be formatted into interactions, user-data, item-data. 
We use [RecSysDatasets - Conversion Tools](https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools) over this pre-processing step. 

