# Benchmarking RoBERTa and Self-Explaining RoBERTa with Data Parallelism 

This submission is based on the **[SelfExplain framework](https://arxiv.org/abs/2103.12279)** and code by **Rajagopal et al. (2021)** 
<br>
## Overview

This project implements benchmarking, profiling and optimizing the performance of RoBERTa and Self-Explain RoBERTa models. We worked on different hardware configurations which includes single GPU and multiple GPUs. We obtained the hardware configuration with better performance and optimized its perfromance using PyTorch DataParallel. 











## Results

| # of GPUs | RoBERTa base | RoBERTa base | Self-Explaining RoBERTa | Self-Explaining RoBERTa |
|-----------|--------------|--------|-------------------------|--------|
|           | CPU time (s) | CUDA time (s) | CPU time (s)           | CUDA time (s) |
| 1         |              |        | 23.715                  | 0.822  |
| 2         |              |        | 22.892                  | 0.851  |
| 4         |              |        | 6.016                   | 0.852  |


## Submission by

This is a submission for **ECE-GY 9143** **High Performance Machine Learning** instructed by **Parijat Dube** at New York University. 

Submission by: <br>**Gauri Gupta** (gg2751) <br>**Navya Sree Nutakki** (nn2382)
