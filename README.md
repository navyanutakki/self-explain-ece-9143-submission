# Benchmarking RoBERTa and Self-Explaining RoBERTa with Data Parallelism 

This submission is based on the **[SelfExplain framework](https://arxiv.org/abs/2103.12279)** and code by **Rajagopal et al. (2021)** 
<br>
## Overview

This project implements benchmarking, profiling and optimizing the performance of RoBERTa and Self-Explain RoBERTa models. We worked on different hardware configurations which includes single GPU and multiple GPUs. We obtained the hardware configuration with better performance and optimized its perfromance using PyTorch DataParallel. 
## Implementation

We initially trained the Self-Explain model on 1 GPU and profiled the training part along with calculating data loading time. We then repeated the same with 2 GPUs and 4 GPUs. Self-Explain performed better with 2 GPUs. There are 16 workers for these variations but now we changed the number of workers to 4 and trained the Self-explain model on 2 GPUs. The model has taken less time to train with 2 GPUs and 4 workers. We then used PyTorch DataParallel to further reduce the time. 
## Code Structure
```

├── ...
├── data # data files for this project
│ ├── RoBERTa-SST-2 # contains SST-2 train and test data along with files created aftre preprocessing
│ 
└── model # model code
│   ├── SE_XLNet.py #self-explain model with LIL and GIL layers
│   ├── data.py     #loads the dataset
│   ├── data_utils.py #pads the matrix with zeros for RoBERTa model
│   ├── infer_model.py #model evaluation
│   ├── model_utils.py 
│   ├── requirements.txt
│
└── outputs
│   ├──
│   ├──
│   ├──
│
└── preprocessing
│   ├── add_ngram_dist.py
│   ├── build_concept_store.py #building the concept store to use with LIL and GIL 
│   ├── constituency_parse.py  #generating the parse tree for the inputs
│   ├── process_trec_dataset.py
│   ├── store_parse_trees.py
│   ├── utils.py
└── scripts
│   ├──profile.sh    #profiling self-explain on 1 GPU and data loading with 16 workers
│   ├──profile_2gpus.sh #profiling self-explain on 2 GPUs and data loading with 16 workers
│   ├──profile_4gpus.sh #profiling self-explain on 2 GPUs and data loading with 16 workers
│   ├──profile_4workers.sh #profiling self-explain on 2 GPUs and data loading with 4 workers
│   ├──profile_4workers_dp.sh #profiling with 2 GPUs, data loading with 4 workers and optimizing with PyTorch DataParallel
│   ├──profile_16workers_dp.sh #profiling with 2 GPUs, data loading with 16 workers and optimizing with PyTorch DataParallel
│   ├──run_preprocessing.sh  #preprocessing the training data
│   ├──run_self_explain.sh #training the model for 5 epochs
└── requirements.txt #contains all the libraries that are required to download

```
## Code Execution

**Self-Explain**

Install all the required libraries as given in the requirements.txt. After downloading train.tsv, test,tsv and dev.tsv from the RoBERTa-SST-2 from data, submit the batch file for preprocessing as sbatch run_preprocessing.sh. After the preproceesing submit a batch file according to the required hardware configurations.

``` shell
pip install -r requirements.txt
```

```
sbatch run_preprocessing.sh
```



## Results

| # of GPUs | RoBERTa base | RoBERTa base | Self-Explaining RoBERTa | Self-Explaining RoBERTa |
|-----------|--------------|--------|-------------------------|--------|
|           | CPU time (s) | CUDA time (s) | CPU time (s)           | CUDA time (s) |
| 1         |      86.98 |  68.703  | 24.002                  | 0.822  |
| 2         |   112.506  | 48.374 | 6.611                  | 0.813  |
| 4         |   87.059   | 68.599 | 6.016                   | 0.852  |


## Submission by

This is a submission for **ECE-GY 9143** **High Performance Machine Learning** instructed by **Parijat Dube** at New York University. 

Submission by: <br>**Gauri Gupta** (gg2751) <br>**Navya Sree Nutakki** (nn2382)
