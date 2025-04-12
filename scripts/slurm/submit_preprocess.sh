#!/bin/bash

domains=('math')

for domain in ${domains[@]}; do
    outfile="dsi_out/$(date +%Y-%m-%d)/preprocess/%x/%j/script.out"
    sbatch --output=$outfile --job-name=$domain-preprocess --export=ALL,STDOUT_PATH=$outfile,DOMAIN=$domain,SCRIPT_PATH=scripts/data_preprocess.sh scripts/slurm/submit.sh
done
