#!/bin/bash

outfile="dsi_out/$(date +%Y-%m-%d)/merge/%j/script.out"
sbatch --output=$outfile --export=ALL,SLURM_JOB_STDOUT_PATH=$outfile src/single_gpu_scripts/merge.sh