#!/bin/bash

#SBATCH --partition=clab,general
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=720:00
#SBATCH --output=dsi_out/%j/script.out
#SBATCH --mem=64G


source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

model_sizes=('160m' '1.4b' '410m')
domains=('math')
lora_ranks=(16 2048 512)
seeds=(1 2 3)
lrs=('2e-3' '2e-4' '2e-2')
wandb offline
wandb disabled

for domain in "${domains[@]}"
do
    for model_size in "${model_sizes[@]}"
    do
        for lora_rank in "${lora_ranks[@]}"
        do
            for lr in "${lrs[@]}"
            do
                for seed in "${seeds[@]}"
                do
                    device_id=$((seed - 1))
                    
                    echo [CUDA_VISIBLE_DEVICES=$device_id] python src/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/lora/pythia-${model_size}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json --pad_sequences  --seed $seed --domain $domain --lr $lr --lora --resume_from_checkpoint

                    CUDA_VISIBLE_DEVICES=$device_id python src/finetune_model.py --base_model EleutherAI/pythia-$model_size --model_config_path configs/model_configs/lora/pythia-${model_size}.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json  --pad_sequences --seed $seed --domain $domain --lr $lr --lora --resume_from_checkpoint >> "dsi_out/${SLURM_JOB_ID}/job_${seed}.out" 2>&1 & 
                done   
                wait
            done
        done
    done
done
