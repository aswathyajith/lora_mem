#!/bin/bash

#SBATCH --partition=clab,general
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --output=dsi_out/%j/script.out
#SBATCH --mem=64G


source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem
cd /net/projects/clab/aswathy/projects/lora_mem

model_sizes=('1.4b')
domains=('legal')
lrs=('2e-4' '2e-6')
wandb offline
wandb disabled
seeds=(1 2 3)
for domain in "${domains[@]}"
do
    for model_size in "${model_sizes[@]}"
    do
        for lr in "${lrs[@]}"
        do 
            for seed in "${seeds[@]}"
            do
                device_id=$((seed - 1))
                echo CUDA_VISIBLE_DEVICES=$device_id python src/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/full-ft/pythia-${model_size}.json --sft_config_path configs/model_configs/sft_configs.json  $lr lr --domain $domain --seed $seed  --lr $lr --pad_sequences --resume_from_checkpoint
                
                CUDA_VISIBLE_DEVICES=$device_id python src/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/full-ft/pythia-${model_size}.json --sft_config_path configs/model_configs/sft_configs.json --domain $domain --seed $seed --lr $lr --pad_sequences --resume_from_checkpoint >> "dsi_out/${SLURM_JOB_ID}/gpu_${device_id}.out" 2>&1 & 
                 
            done
            wait
        done
    done
done