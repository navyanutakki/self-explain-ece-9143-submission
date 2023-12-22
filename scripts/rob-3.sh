#!/bin/bash
#SBATCH --job-name=rob-3
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4


singularity exec --nv --overlay /scratch/gg2751/hproject/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; \
    python run-model.py"