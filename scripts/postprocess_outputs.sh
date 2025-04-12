#!/bin/bash

source /net/projects/clab/aswathy/projects/lora_mem/scripts/env_setup.sh

domain=$1
domain_datasets=$2
n_train_tkns=("2e4" "2e5" "2e6")
max_seq_lens=(64 128 256)
split="train"

for domain_dataset in ${domain_datasets[@]}; do
    for model in ${model_sizes[@]}; do
        for num_train in ${num_trains[@]}; do
            for max_seq_len in ${max_seq_lens[@]}; do
                data_dir="${domain_dataset}/$split/num_train_${num_train}/max_seq_len_${max_seq_len}/sample_${sample_size}"
                for seed in ${seeds[@]}; do
                    echo python src/postprocess_outputs.py --data_merge_config_path configs/postprocess_configs/pythia-1.4b.json --output_dir data/output_token_info/$model/packing/perturbations/reverse_tkns --seed $seed --data_dir $data_dir

                    stdout_file="$SLURM_JOB_OUTPUT_DIR/logs-ntrain_${num_train}-max_seq_len_${max_seq_len}-seed_${seed}.out"

                    python src/postprocess_outputs.py --postprocess_config configs/postprocess_configs/pythia-1.4b.json --output_dir data/output_token_info/$model/packing/perturbations/reverse_tkns --seed $seed --data_dir $data_dir >> $stdout_file 2>&1
                done
            done
        done
    done
done