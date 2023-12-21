# Benchmarking RoBERTa and Self-Explaining RoBERTa with Data Parallelism 

This submission is based on the **[SelfExplain framework](https://arxiv.org/abs/2103.12279)** and code by **Rajagopal et al. (2021)** 
<br>
## Overview

This project implements benchmarking, profiling and optimizing the performance of RoBERTa and Self-Explain RoBERTa models. We worked on different hardware configurations which includes single GPU and multiple GPUs. We obtained the hardware configuration with better performance and optimized its perfromance using PyTorch DataParallel. 
## Implementation

We initially trained the Self-Explain model on 1 GPU and profiled the training part along with calculating data loading time. We then repeated the same with 2 GPUs and 4 GPUs. Self-Explain performed better with 2 GPUs. There are 16 workers for these variations but now we changed the number of workers to 4 and trained the Self-explain model on 2 GPUs. The model has taken less time to train with 2 GPUs and 4 workers. We then used PyTorch DataParallel to further reduce the time. We also used PyTorch DataParallel on 2 GPUs and 16 workers but 2 GPUs and 4 workers has given better performance with DataParallel.
## Code Structure
```

├── ...
├── data # data files for this project
│   ├── RoBERTa-SST-2 # contains SST-2 train and test data along with files created after preprocessing
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
│   ├── profile.out
│   ├── profile_16workers_dp.out
│   ├── profile_2gpus.out
│   ├── profile_4gpus.out
│   ├── profile_4workers.out
│   ├── profile_4workers_dp.out
│   ├── roberta-base.out
│
└── preprocessing
│   ├── add_ngram_dist.py
│   ├── build_concept_store.py #building the concept store to use with LIL and GIL 
│   ├── constituency_parse.py  #generating the parse tree for the inputs
│   ├── process_trec_dataset.py
│   ├── store_parse_trees.py
│   ├── utils.py
│
└── roberta-base
│   ├── data_sst2.py
│   ├── models.py 
│   ├── run_Roberta_model.py  
│   ├── utils.py
│
└── scripts
│   ├──profile.sh    #profiling self-explain on 1 GPU and data loading with 16 workers
│   ├──profile_2gpus.sh #profiling self-explain on 2 GPUs and data loading with 16 workers
│   ├──profile_4gpus.sh #profiling self-explain on 2 GPUs and data loading with 16 workers
│   ├──profile_4workers.sh #profiling self-explain on 2 GPUs and data loading with 4 workers
│   ├──profile_4workers_dp.sh #profiling with 2 GPUs, data loading with 4 workers and optimizing with PyTorch DataParallel
│   ├──profile_16workers_dp.sh #profiling with 2 GPUs, data loading with 16 workers and optimizing with PyTorch DataParallel
│   ├──roberta.sh #profiling the RoBERTa base on 4 GPUs
│   ├──run_preprocessing.sh  #preprocessing the training data
│   ├──run_self_explain.sh #training the model for 5 epochs
│
└── requirements.txt #contains all the libraries that are required to download

```
## Code Execution

**Self-Explain**

Install all the required libraries as given in the requirements.txt. After downloading train.tsv, test,tsv and dev.tsv from the RoBERTa-SST-2 from data, submit the batch file for preprocessing as sbatch run_preprocessing.sh. After the preprocessing submit a batch file according to the required hardware configurations.

**RoBERTa**

Submit the roberta.sh batch file which has 4 GPUs. To run the RoBERTa base on 1 or 2 GPUs, change the number of GPUs respectively.

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

CPU time and CUDA time are higher for RoBERTa base wtih 4 GPUs when compared to Self-Explain model. From the profiling results we observed that aten::mm and aten::mm in RoBERTa-base are significantly taking higher CUDA time 28.593s and 20.248s respectively whereas they are 1.191ms and 63.660ms respectively in Self-Explain with 4 GPUs.

## Submission by

This is a submission for **ECE-GY 9143** **High Performance Machine Learning** instructed by **Parijat Dube** at New York University. 

Submission by: <br>**Gauri Gupta** (gg2751) <br>**Navya Sree Nutakki** (nn2382)
