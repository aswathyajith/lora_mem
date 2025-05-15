#!/bin/bash

domains=('math' 'code' 'legal')
target_modules=('all-linear' 'attn_only')
model_sizes=('160m' '410m' '1.4b')

for model_size in ${model_sizes[@]}; do
    for domain in ${domains[@]}; do
        outfile="dsi_out/$(date +%Y-%m-%d)/finetuning/full/%x/%j/script.out"
        sbatch --output=$outfile --job-name=$domain-full --export=ALL,STDOUT_PATH=$outfile,DOMAIN=$domain,MODEL_SIZE=$model_size,SCRIPT_PATH=scripts/full_finetuning.sh scripts/slurm/run_script.sh
        for target_module in ${target_modules[@]}; do
            outfile="dsi_out/$(date +%Y-%m-%d)/finetuning/lora/%x/%j/script.out"
            sbatch --output=$outfile --job-name=$domain-lora-${target_module} --export=ALL,STDOUT_PATH=$outfile,DOMAIN=$domain,TARGET_MODULE=$target_module,MODEL_SIZE=$model_size,SCRIPT_PATH=scripts/lora_finetuning.sh scripts/slurm/run_script.sh
        done
    done
done

