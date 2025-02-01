#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=16G
#SBATCH --nodelist=j003-ds


source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

MODEL_DIR='/net/projects/clab/aswathy/projects/lora_mem/'
model_sizes=('1.4b')
num_trains=(4096)
lr='2e-6'
batch_sizes=(128)
dataset_path="" # "hellaswag/"
dataset_name="wikitext:wikitext-2-raw-v1" #"Rowan/hellaswag"
es="early_stopping/"

for model_size in "${model_sizes[@]}"
do
    for num_train in "${num_trains[@]}"
    do
        for batch_size in "${batch_sizes[@]}"
        do  
        # python src/model_generations.py --model_ckpts_path EleutherAI/pythia-$model_size --lora_ckpts_path models/lora/pythia-$model_size/r_16/lr_2e-4/num_train_$num_train --results_path results/pythia-$model_size/lora/r_16/lr_2e-4/num_train_$num_train/generations/ --num_train $num_train
        
            python src/model_generations.py --model_ckpts_path ${MODEL_DIR}models/${dataset_path}full-ft/pythia-$model_size/lr_${lr}/${es}num_train_$num_train/bsize_$batch_size --results_path results/pythia-$model_size/${dataset_path}full-ft/lr_${lr}/${es}num_train_$num_train/bsize_${batch_size}/generations/ --num_train $num_train --dataset $dataset_name --max_seq_len 128 #--pad_sequences
        done
    done
done
