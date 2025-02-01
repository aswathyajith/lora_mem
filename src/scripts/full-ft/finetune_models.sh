#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=16G


source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

model_sizes=('1.4b')
num_trains=(4096)
batch_sizes=(128)
max_seq_len=128
dataset="wikitext:wikitext-2-raw-v1"
lr='2e-6'

wandb offline
wandb disabled

for model_size in "${model_sizes[@]}"
do
    
    for num_train in "${num_trains[@]}"
    do
        for batch_size in "${batch_sizes[@]}"
        do
            echo finetuning LR=$lr python src/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/pythia-${model_size}-full-ft/pythia-${model_size}_${num_train}_b${batch_size}.json --sft_config_path configs/model_configs/sft_configs.json --max_seq_len $max_seq_len --dataset $dataset --pad_sequences --num_train $num_train 
            
            python src/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/pythia-${model_size}-full-ft/pythia-${model_size}_${num_train}_b${batch_size}.json --sft_config_path configs/model_configs/sft_configs.json --max_seq_len $max_seq_len --dataset $dataset --pad_sequences --num_train $num_train 
        done
    done
done