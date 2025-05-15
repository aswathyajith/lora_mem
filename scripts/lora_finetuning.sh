#!/bin/bash

source /net/projects/clab/aswathy/projects/lora_mem/scripts/env_setup.sh

MODEL_SIZE=$1
TARGET_MODULE=$2

# check if MODEL_SIZE is set
if [ -z "${MODEL_SIZE}" ] || [ -z "${TARGET_MODULE}" ]; then
    echo "MODEL_SIZE or TARGET_MODULE is not set"
    exit 1
fi

# function to get array of ranks for given model size
get_ranks() {
    if [ $MODEL_SIZE == '160m' ]; then
        ranks=(768 384 192 96 48 24 12 6 3 1)
    elif [ $MODEL_SIZE == '410m' ]; then
        ranks=(1024 512 256 128 64 32 16 8 4 2 1)
    elif [ $MODEL_SIZE == '1.4b' ]; then
        ranks=(2048 1024 512 256 128 64 32 16 8 4 2 1)
    fi
    echo ${ranks[@]}
}

lora_ranks=($(get_ranks $MODEL_SIZE))
echo "LoRA_RANKS: ${lora_ranks[@]}"

if [ -z "${STDOUT_PATH}" ]; then
    STDOUT_PATH="logs"
fi

echo "STDOUT_PATH is set to $STDOUT_PATH"
seeds=(1 2 3)
lrs=('2e-5' '2e-4' '2e-3')
n_train_tkns=('2e6')

for seed in "${seeds[@]}"
    do
    for lr in "${lrs[@]}"
        do
        for lora_rank in "${lora_ranks[@]}"
        do
            for n_tkns in "${n_train_tkns[@]}"
            do      
                echo $n_tkns
                stdout_file="$STDOUT_PATH/log_domain-${DOMAIN}_model-${MODEL_SIZE}_lora-r${lora_rank}_lr-${lr}_n_train_tkns-${n_tkns}_seed${seed}.out"
                echo $stdout_file

                echo "python src/finetuning/finetune_model.py --base_model EleutherAI/pythia-$MODEL_SIZE --model_config_path configs/model_configs/lora/pythia-${MODEL_SIZE}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json  --packing --seed $seed --domain $DOMAIN --lr $lr --lora --lora_rank $lora_rank --dataset_config configs/dataset_config.json --resume_from_checkpoint --n_train_tkns $n_tkns --target_modules $TARGET_MODULE >> $stdout_file 2>&1"

                python src/finetuning/finetune_model.py --base_model EleutherAI/pythia-$MODEL_SIZE --model_config_path configs/model_configs/lora/pythia-${MODEL_SIZE}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json  --packing --seed $seed --domain $DOMAIN --lr $lr --lora --lora_rank $lora_rank --dataset_config configs/dataset_config.json --resume_from_checkpoint --n_train_tkns $n_tkns --target_modules $TARGET_MODULE >> $stdout_file 2>&1  

                echo "Exit code: $?"
            done
        done
    done
done