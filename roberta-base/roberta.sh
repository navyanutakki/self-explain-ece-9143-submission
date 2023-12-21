#!/bin/bash
#SBATCH --job-name=roberta-base
#SBATCH --output=%x.out
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4


singularity exec --nv --overlay /scratch/nn2382/my_env/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; \
    python roberta/run_Roberta_model.py"
