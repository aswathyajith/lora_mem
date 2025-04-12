#!/bin/bash

source /net/projects/clab/aswathy/projects/lora_mem/scripts/env_setup.sh

model_sizes=("pythia-1.4b")
seeds=(1 2 3)
domain=$1
split="validation"
n_train_tkns=(2e4 2e5 2e6)

for model_size in "${model_sizes[@]}"; do
    MODEL_DIR="models/${model_size}/packing/perturbations/none"
    MODEL_OUTPUTS_DIR="results/model_gens/${model_size}/packing/perturbations/none"
    for domain in "${domains[@]}"; do
        for seed in "${seeds[@]}"; do
            for n_tkns in "${n_train_tkns[@]}"; do
                OUTPUT_PATH="${SLURM_JOB_OUTPUT_DIR}/logs_${model_size}_domain_${domain}_n_tkns_${n_tkns}_seed_${seed}.out"
                
                echo "python src/token_freq_probs.py --base_model EleutherAI/${model_size} --domain $domain --split $split --model_dir $MODEL_DIR --model_outputs_dir $MODEL_OUTPUTS_DIR --n_train_tkns $n_tkns --seed $seed --skip_existing --packing >> ${OUTPUT_PATH} 2>&1"

                python src/token_freq_probs.py --base_model EleutherAI/${model_size} --domain $domain --split $split --model_dir $MODEL_DIR --model_outputs_dir $MODEL_OUTPUTS_DIR --n_train_tkns $n_tkns --seed $seed --skip_existing --packing >> ${OUTPUT_PATH} 2>&1 # & 

                # print exit code
                echo "Exit code: $?"
            done
        done
    done
done