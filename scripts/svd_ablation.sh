#!/bin/bash

source /net/projects/clab/aswathy/projects/lora_mem/scripts/env_setup.sh
model_sizes=('pythia-1.4b')
ranks=(1 4 16 128 1024 2048)
seeds=(1)
max_seq_lens=(256)
n_tkns_set=('2e6')

if [ -z "${STDOUT_PATH}" ]; then
    STDOUT_PATH="logs"
fi

echo "STDOUT_PATH is set to $STDOUT_PATH"

if [ -z "${DOMAIN}" ]; then
    echo "Domain is not set. Exiting..."
    exit 1
fi

if [ -z "${TARGET_MODULE}" ]; then
    echo "Target module is not set. Exiting..."
    exit 1
fi
echo "Domain is set to $DOMAIN"
echo "Target module is set to $TARGET_MODULE"
config_path="configs/model_data_train_config.csv"

for model_size in ${model_sizes[@]}; do
    for max_seq_len in ${max_seq_lens[@]}; do
        for n_tkns in ${n_tkns_set[@]}; do
            for seed in ${seeds[@]}; do
                stdout_file="$STDOUT_PATH/log_domain-${DOMAIN}_model-${model_size}_svd_ablation_seed${seed}.out"
                
                exec_cmd="python src/ablations/svd_full_model.py --config_path $config_path --model_size $model_size --domain $DOMAIN --target_module $TARGET_MODULE --ranks ${ranks[@]} --skip_existing >> $stdout_file 2>&1"

                echo "Evaluating models on train data..."
                echo $exec_cmd
                eval $exec_cmd

                echo "Exit code: $?"
            done
        done
    done
done