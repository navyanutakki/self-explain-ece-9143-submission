#!/bin/bash
#SBATCH --job-name=project
#SBATCH --output=%x.out
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4



#export TOKENIZERS_PARALLELISM=false
#python model/run.py --dataset_basedir data/XLNet-SUBJ \
#                         --lr 2e-5  --max_epochs 5 \
#                         --gpus 1 \
#                         --concept_store data/XLNet-SUBJ/concept_store.pt \
#                         --accelerator ddp \
#                         --gamma 0.1 \
#                        --lamda 0.1 \
#                         --topk 5

# for RoBERTa

singularity exec --nv --overlay /scratch/nn2382/my_env/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; \
    python roberta/run_Roberta_model.py"