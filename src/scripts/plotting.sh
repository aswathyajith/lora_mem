#!/bin/bash

source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

model_sizes=('160m' '410m' '1.4b')
model_sizes=('1.4b')
num_trains=(4096)
batch_size=128
dataset="hellaswag/"
early_stopping="early_stopping/"

for model_size in "${model_sizes[@]}"
do
    for num_train in "${num_trains[@]}"
    do
        # lora
        lr='2e-4'
        # python src/plot_mem_in_training.py --exact_mem_data_path results/pythia-$model_size/lora/r_16/lr_$lr/num_train_$num_train/mem_at_len_k.jsonl --exact_mem_plt_path results/plots/pythia-$model_size/lora/lr_$lr/num_train_$num_train/exact_mem.png

        # python src/plot_mem_in_training.py --partial_mem_data_path results/pythia-$model_size/${dataset}lora/r_16/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/partial_mem.csv --loss_path results/pythia-${model_size}/${dataset}lora/r_16/lr_${lr}/${early_stopping}num_train_${num_train}/bsize_${batch_size}/loss.csv --lcs_plt_path results/plots/pythia-$model_size/${dataset}/lora/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/partial_lcs.png --substr_plt_path results/plots/pythia-$model_size/${dataset}/lora/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/partial_max_lcsub.png --early_stopping

        # full ft
        lr='2e-6'
        # python src/plot_mem_in_training.py --exact_mem_data_path results/pythia-$model_size/full-ft/num_train_$num_train/mem_at_len_k.jsonl --exact_mem_plt_path results/plots/pythia-$model_size/full-ft/num_train_$num_train/exact_mem.png

        python src/plot_mem_in_training.py --partial_mem_data_path results/pythia-$model_size/${dataset}full-ft/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/partial_mem.csv --loss_path results/pythia-$model_size/${dataset}full-ft/lr_$lr/${early_stopping}num_train_$num_train/bsize_${batch_size}/loss.csv --lcs_plt_path results/plots/pythia-$model_size/${dataset}full-ft/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/partial_lcs.png --substr_plt_path results/plots/pythia-$model_size/${dataset}full-ft/lr_$lr/${early_stopping}num_train_$num_train/bsize_$batch_size/partial_max_lcsub.png --early_stopping
    done
done

