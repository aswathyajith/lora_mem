#!/bin/bash

source /net/projects/clab/aswathy/projects/lora_mem/scripts/env_setup.sh

seq_lens=(64 128 256)
lora_rank=16
domain_datasets=$1
split="train"
seeds=(1 2 3)
for seed in ${seeds[@]}; do
    for domain_dataset in ${domain_datasets[@]}; do
        if [ $domain_dataset == "biomed/chemprot" ]; then
            num_trains=("all")
            sample_size=("sample_all")
        else
            sample_size=("sample_2048")
            num_trains=("4096" "8192" "16384" "all")
            # if [ $lora_rank == 256 ]; then
            #     num_trains=("all")
            # else
            #     num_trains=("4096" "8192" "16384" "all")
            # fi
        fi
        
        for num_train in ${num_trains[@]}; do
            for seq_len in ${seq_lens[@]}; do
                python src/merge_n_samples.py --data_path data/output_token_info/pythia-1.4b/packing/perturbations/none/$domain_dataset/$split/num_train_${num_train}/max_seq_len_${seq_len}/$sample_size --output_dir data/plotting_data/pythia-1.4b/packing/perturbations/none/$domain_dataset/$split/num_train_${num_train}/max_seq_len_${seq_len}/$sample_size --model1_seed $seed --model2_seed $seed --model1_r $lora_rank
            done
        done
    done
done