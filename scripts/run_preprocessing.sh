#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

export DATA_FOLDER='data/RoBERTa-SST-2'
export TOKENIZER_NAME='roberta-base'
export MAX_LENGTH=5

# Creates jsonl files for train and dev

singularity exec --nv \
--overlay /scratch/nn2382/my_env/overlay-25GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; python preprocessing/store_parse_trees.py \
      --data_dir $DATA_FOLDER  \
      --tokenizer_name $TOKENIZER_NAME"

# Create concept store for SST-2 dataset
# Since SST-2 already provides parsed output, easier to do it this way, for other datasets, need to adapt
singularity exec --nv \
--overlay /scratch/nn2382/my_env/overlay-25GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; python preprocessing/build_concept_store.py \
       -i $DATA_FOLDER/train_with_parse.json \
       -o $DATA_FOLDER \
       -m $TOKENIZER_NAME \
       -l $MAX_LENGTH"

