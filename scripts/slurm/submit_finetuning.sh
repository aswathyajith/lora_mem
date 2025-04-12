#!/bin/bash

# domains=('wiki' 'bible' 'biomed' 'legal' 'math' 'code')
domains=('math')
for domain in ${domains[@]}; do
    outfile="dsi_out/$(date +%Y-%m-%d)/finetuning/full/%x/%j/script.out"
    sbatch --output=$outfile --job-name=$domain-full --export=ALL,STDOUT_PATH=$outfile,DOMAIN=$domain,SCRIPT_PATH=scripts/full_finetuning.sh scripts/slurm/submit.sh

    outfile="dsi_out/$(date +%Y-%m-%d)/finetuning/lora/%x/%j/script.out"
    sbatch --output=$outfile --job-name=$domain-lora --export=ALL,STDOUT_PATH=$outfile,DOMAIN=$domain,SCRIPT_PATH=scripts/lora_finetuning.sh scripts/slurm/submit.sh
done

