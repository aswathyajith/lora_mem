#!/bin/bash

source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem
cd /net/projects/clab/aswathy/projects/lora_mem
wandb offline
wandb disabled

# export HF_DATASETS_CACHE=/net/projects/clab/aswathy/projects/lora_mem/hf_cache
# export HF_HOME=/net/projects/clab/aswathy/projects/lora_mem/hf_home
