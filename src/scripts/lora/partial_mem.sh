#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:600:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=16G

source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate test_env

model_sizes=('1.4b')
num_trains=(4096)
lr='2e-4'
batch_sizes=(128)
dataset="" #dataset="hellaswag/"
early_stopping="early_stopping/"

for model_size in "${model_sizes[@]}"
do
    for num_train in "${num_trains[@]}"
    do
        for batch_size in "${batch_sizes[@]}"
        do 

            echo python src/measure_partial_mem.py --model_gen_path results/pythia-$model_size/${dataset}lora/r_16/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/generations/ --batch_size $batch_size --num_train $num_train
            python src/measure_partial_mem.py --model_gen_path results/pythia-$model_size/${dataset}lora/r_16/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/generations/ --batch_size $batch_size --num_train $num_train
        done
    done
done