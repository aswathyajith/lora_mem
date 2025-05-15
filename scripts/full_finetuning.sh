#!/bin/bash

source /net/projects/clab/aswathy/projects/lora_mem/scripts/env_setup.sh 

# check if MODEL_SIZE and DOMAIN are set
if [ -z "${MODEL_SIZE}" ] || [ -z "${DOMAIN}" ]; then
    echo "MODEL_SIZE or DOMAIN is not set"
    exit 1
fi


lrs=('2e-7' '2e-6' '2e-5')
n_train_tkns=('2e6')
seeds=(1 2 3)
stream_sizes=('10000')

if [ -z "${STDOUT_PATH}" ]; then
    STDOUT_PATH="logs"
fi

echo "STDOUT_PATH is set to $STDOUT_PATH"



for lr in "${lrs[@]}"
do 
    for n_tkns in "${n_train_tkns[@]}"
    do
        for seed in "${seeds[@]}"
        do
            
            stdout_file="$STDOUT_PATH/log_domain-${DOMAIN}_model-${MODEL_SIZE}_full_lr-${lr}_n_train_tkns-${n_tkns}_seed${seed}.out"
            
            echo "python src/finetuning/finetune_model.py --base_model EleutherAI/pythia-$MODEL_SIZE --model_config_path configs/model_configs/full-ft/pythia-${MODEL_SIZE}.json --sft_config_path configs/model_configs/sft_configs.json --packing --seed $seed --domain $DOMAIN --lr $lr --dataset_config configs/dataset_config.json  --resume_from_checkpoint --n_train_tkns $n_tkns >> $stdout_file 2>&1"

            python src/finetuning/finetune_model.py --base_model EleutherAI/pythia-$MODEL_SIZE --model_config_path configs/model_configs/full-ft/pythia-${MODEL_SIZE}.json --sft_config_path configs/model_configs/sft_configs.json --packing --seed $seed --domain $DOMAIN --lr $lr --dataset_config configs/dataset_config.json  --resume_from_checkpoint --n_train_tkns $n_tkns >> $stdout_file 2>&1

            echo "Exit code: $?"
        done
    done
done

