#!/bin/bash

domains=('legal' 'math' 'code')
for domain in ${domains[@]}; do
    outfile="dsi_out/$(date +%Y-%m-%d)/select_models/%x/%j/script.out"
    sbatch --output=$outfile --job-name=$domain-select-models --export=ALL,STDOUT_PATH=$outfile,DOMAIN=$domain,SCRIPT_PATH=scripts/select_models.sh scripts/slurm/run_script.sh 
done

