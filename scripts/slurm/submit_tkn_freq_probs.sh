#!/bin/bash

outfile="dsi_out/$(date +%Y-%m-%d)/token_freq_probs/%x/%j/script.out"
sbatch --output=$outfile --export=ALL,SLURM_JOB_STDOUT_PATH=$outfile src/single_gpu_scripts/tkn_freq_probs_wiki.sh
sbatch --output=$outfile --export=ALL,SLURM_JOB_STDOUT_PATH=$outfile src/single_gpu_scripts/tkn_freq_probs_biomed.sh
sbatch --output=$outfile --export=ALL,SLURM_JOB_STDOUT_PATH=$outfile src/single_gpu_scripts/tkn_freq_probs_bible.sh