#!/bin/bash

#SBATCH --partition=general,clab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=720:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=16G


source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

model_sizes=('1.4b')
num_trains=(4096)
bsizes=(128)
max_seq_len=128
dataset='wikitext:wikitext-2-raw-v1'
lr='2e-4'

wandb offline
wandb disabled

for model_size in "${model_sizes[@]}"
do
    
    for num_train in "${num_trains[@]}"
    do
        for bsize in "${bsizes[@]}"
        do
            seeds=(1)
            for seed in "${seeds[@]}"
            do
                echo finetuning LR=$lr python src/finetune_model.py --run_name "EleutherAI/pythia-$model_size" --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/pythia-${model_size}-lora/pythia-${model_size}_${num_train}_b${bsize}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json --dataset $dataset --max_seq_len $max_seq_len --pad_sequences --lora --num_train $num_train --seed $seed
                python src/finetune_model.py --run_name "EleutherAI/pythia-$model_size" --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/pythia-${model_size}-lora/pythia-${model_size}_${num_train}_b${bsize}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json --dataset $dataset --max_seq_len $max_seq_len --pad_sequences --lora --num_train $num_train --seed $seed
            done
        done
    done
done
