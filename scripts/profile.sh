#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --output=%x.out
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1



export TOKENIZERS_PARALLELISM=false
#python model/run.py --dataset_basedir data/XLNet-SUBJ \
#                         --lr 2e-5  --max_epochs 5 \
#                         --gpus 1 \
#                         --concept_store data/XLNet-SUBJ/concept_store.pt \
#                         --accelerator ddp \
#                         --gamma 0.1 \
#                        --lamda 0.1 \
#                         --topk 5

# for RoBERTa

singularity exec --nv --overlay /scratch/$USER/my_env/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; \
    python model/run.py --dataset_basedir data/RoBERTa-SST-2 \
        --lr 2e-5 --max_epochs 5 --gpus 1 \
        --concept_store data/RoBERTa-SST-2/concept_store.pt \
        --accelerator cuda --model_name roberta-base --topk 5 --gamma 0.1 --lamda 0.1"
