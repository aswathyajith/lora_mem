#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=dsi_out/%j.out
#SBATCH --mem=16G

python src/next_token_probs.py --base_model EleutherAI/pythia-1.4b --ckpt_path models/lora/pythia-1.4b/r_1024/lr_2e-4/early_stopping/num_train_4096/bsize_128 --output_dir_path results/pythia-1.4b/lora/r_1024/lr_2e-4/early_stopping/val/bsize_128