# Benchmarking RoBERTa and Self-Explaining RoBERTa with Data Parallelism 

This submission is based on the [SelfExplain framework](https://arxiv.org/abs/2103.12279) and code by Rajagopal et al. (2021) 

## Usage

```shell
mkdir /scratch/$USER/myenv
cd /scratch/$USER/myenv
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
```
```shell
cd /scratch/$USER/
```
```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
```

```shell
touch /ext3/env.sh
echo '#!/bin/bash' >> /ext3/env.sh
echo 'source /ext3/miniconda3/etc/profile.d/conda.sh' >> /ext3/env.sh
echo 'export PATH=/ext3/miniconda3/bin:$PATH'         >> /ext3/env.sh
echo 'export PYTHONPATH=/ext3/miniconda3/bin:$PATH'   >> /ext3/env.sh
```

```shell
source /ext3/env.sh
conda update -n base conda -y
conda clean --all --yes
conda install pip --yes
conda install ipykernel --yes
exit
```

```shell
srun --cpus-per-task=2 --mem=10GB --time=02:00:00 --pty /bin/bash
singularity exec --overlay /scratch/$USER/myenv/overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash
source /ext3/env.sh
```

```shell
pip install -r requirements.txt
python
>>> import benepar
>>> benepar.download('benepar_en3')
```


```shell
sh scripts/run_preprocessing.sh
```

```shell
sh scripts/run_self_explain.sh
```

## Results

| # of GPUs | RoBERTa base | RoBERTa base | Self-Explaining RoBERTa | Self-Explaining RoBERTa |
|-----------|--------------|--------|-------------------------|--------|
|           | CPU time (s) | CUDA time (s) | CPU time (s)           | CUDA time (s) |
| 1         |              |        | 23.715                  | 0.822  |
| 2         |              |        | 22.892                  | 0.851  |
| 4         |              |        | 6.016                   | 0.852  |


## Submission by

This is a submission for **ECE-GY 9143** **High Performance Machine Learning** instructed by **Parijat Dube** at New York University. 

Submission by: <br>Gauri Gupta (gg2751) <br>Navya Sree Nutakki (nn2382)
