#!/bin/bash

#SBATCH --partition=clab,general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=dsi_out/%j/script.out
#SBATCH --mem=64G

source /home/aswathy/miniconda3/etc/profile.d/conda.sh
conda activate lora_mem

model_sizes=("pythia-160m" "pythia-410m" "pythia-1.4b")
model_sizes=("pythia-1.4b")
seeds=(1 2 3)
max_length=2048
lora_ranks=(2048)
domains=("legal")
split="train"
# python src/token_freq_probs.py --base_model EleutherAI/${model_size} --max_length $max_length --freq_save_dir $freq_save_dir --shuffle --pretraining_corpus --skip_existing


for model_size in "${model_sizes[@]}"; do
    MODEL_DIR="models/${model_size}"
    for domain in "${domains[@]}"; do
        for seed in "${seeds[@]}"; do
                # python src/token_freq_probs.py --split train --base_model EleutherAI/${model_size} --base_save_path results/${model_size}/${data_domain}/train/sample_${num_train}/base/tkn_freq_probs.csv --full_model_path ${FULL_MODEL_DIR}/seed_${seed}/final_model --full_save_path results/${model_size}/${data_domain}/train/sample_${num_train}/full-ft/lr_${lr_full_ft}/seed_${seed}/tkn_freq_probs.csv --lora_adapter_path ${LORA_MODEL_DIR}/seed_${seed}/final_model --lora_save_path results/${model_size}/${data_domain}/train/sample_${num_train}/lora/r_${lora_rank}/lr_${lr_lora}/seed_${seed}/tkn_freq_probs.csv --num_train $num_train --dataset $dataset --max_length $max_length --shuffle --skip_existing
                
                CUDA_VISIBLE_DEVICES=0 #$((seed-1))
                echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/token_freq_probs.py --base_model EleutherAI/${model_size} --domain $domain --split $split --model_dir $MODEL_DIR --model_outputs_dir results/model_gens/${model_size} --seed $seed
                CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python src/token_freq_probs.py --base_model EleutherAI/${model_size} --domain $domain --split $split --model_dir $MODEL_DIR --model_outputs_dir results/model_gens/${model_size} --seed $seed >> dsi_out/${SLURM_JOB_ID}/gpu_${seed}.out 2>&1 # & 
        done
        wait
    done
done

