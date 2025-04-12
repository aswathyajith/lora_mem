#!/bin/bash

source /net/projects/clab/aswathy/projects/lora_mem/scripts/env_setup.sh

model_sizes=('1.4b')
domain=$DOMAIN
lrs=('2e-7' '2e-6' '2e-5')
n_train_tkns=('2e4' '2e5' '2e6')
seeds=(1 2 3)
lora_ranks=(16)

if [ -z "${STDOUT_PATH}" ]; then
    STDOUT_PATH="logs"
fi

echo "STDOUT_PATH is set to $STDOUT_PATH"

for model_size in "${model_sizes[@]}"
do
    for lora_rank in "${lora_ranks[@]}"
    do
        for lr in "${lrs[@]}"
        do
            for n_tkns in "${n_train_tkns[@]}"
            do 
                for seed in "${seeds[@]}"
                do
                    
                    stdout_file="$STDOUT_PATH/log_domain-${domain}_model-${model_size}_lora-r${lora_rank}_lr-${lr}_n_train_tkns-${n_tkns}_seed${seed}.out"

                    echo "python src/training/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/lora/pythia-${model_size}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json  --packing --seed $seed --domain $domain --lr $lr --lora --lora_rank $lora_rank --dataset_config configs/dataset_config.json --resume_from_checkpoint --n_train_tkns $n_tkns >> $stdout_file 2>&1"

                    python src/training/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/lora/pythia-${model_size}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json  --packing --seed $seed --domain $domain --lr $lr --lora --lora_rank $lora_rank --dataset_config configs/dataset_config.json --resume_from_checkpoint --n_train_tkns $n_tkns >> $stdout_file 2>&1  

                    echo "Exit code: $?"
                done
            done
        done
    done
done