#!/bin/bash

outfile="dsi_out/$(date +%Y-%m-%d)/postprocess_outputs/%x/%j/script.out"
sbatch --output=$outfile --export=ALL,SLURM_JOB_STDOUT_PATH=$outfile src/single_gpu_scripts/postprocess_outputs_wiki.sh
# sbatch --output=$outfile --export=ALL,SLURM_JOB_STDOUT_PATH=$outfile src/single_gpu_scripts/postprocess_outputs_bio.sh
# sbatch --output=$outfile --export=ALL,SLURM_JOB_STDOUT_PATH=$outfile src/single_gpu_scripts/postprocess_outputs_bible.sh