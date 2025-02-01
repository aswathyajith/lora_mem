#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:300:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=16G

source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

lr='2e-6'
num_train=4096
model_size='1.4b'
bsize=128
# dataset_path="hellaswag/"
# dataset="Rowan/hellaswag"
dataset_path=""
dataset="wikitext:wikitext-2-raw-v1"
es="early_stopping/"
max_length=128

echo Full-FT, LR: $lr
python src/eval_model_ckpts.py --model_ckpts_path models/${dataset_path}full-ft/pythia-$model_size/lr_$lr/${es}num_train_$num_train/bsize_$bsize --model_config_path configs/model_configs/pythia-${model_size}-full-ft/pythia-${model_size}_${num_train}_b${bsize}.json --results_path results/pythia-${model_size}/${dataset_path}full-ft/lr_${lr}/${es}num_train_$num_train/bsize_$bsize/loss.csv --dataset $dataset --batch_size $bsize --max_length $max_length #--compute_acc --pad_sequences