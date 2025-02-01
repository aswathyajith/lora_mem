#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:720:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=16G

source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

MODEL_DIR='/net/projects/clab/aswathy/projects/lora_mem/'
model_sizes=('1.4b')
num_trains=(4096)
lr='2e-4'
batch_sizes=(128)
dataset_path="" #"hellaswag/"
dataset_name="wikitext:wikitext-2-raw-v1"
es="early_stopping/"

for model_size in "${model_sizes[@]}"
do
    for num_train in "${num_trains[@]}"
    do
        for batch_size in "${batch_sizes[@]}"
        do  
            python src/model_generations.py --model_ckpts_path EleutherAI/pythia-$model_size --lora_ckpts_path models/${dataset_path}lora/pythia-$model_size/r_16/lr_$lr/${es}num_train_$num_train/bsize_$batch_size --results_path results/pythia-$model_size/${dataset_path}lora/r_16/lr_$lr/${es}num_train_$num_train/bsize_${batch_size}/generations/ --num_train $num_train --dataset $dataset_name --max_seq_len 128 #--pad_sequences

            # python src/model_generations.py --model_ckpts_path models/full-ft/pythia-$model_size/num_train_$num_train --results_path results/pythia-$model_size/full-ft/num_train_$num_train/generations/ --num_train $num_train
        done

    done
done
