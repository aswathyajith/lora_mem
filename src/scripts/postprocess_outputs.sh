#!/bin/bash

#SBATCH --partition=clab,general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=64G

source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

model_sizes=("pythia-160m" "pythia-410m" "pythia-1.4b")
model_sizes=("pythia-1.4b")

python src/postprocess_outputs.py 