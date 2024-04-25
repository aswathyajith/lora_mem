#!/bin/bash

context_lens=(25)

# # pretrained
# models=("EleutherAI/pythia-160m" "EleutherAI/pythia-410m" "EleutherAI/pythia-1.4b")
# output_path_prefix=("results/pythia-160m/pretrained/epoch_20/train_set_completions/" "results/pythia-410m/pretrained/epoch_20/train_set_completions/" "results/pythia-1.4b/pretrained/epoch_20/train_set_completions/")

# for i in "${!models[@]}"
# do 
    
#     model=${models[i]}
#     output_path_pref=${output_path_prefix[i]}
#     # echo ${model} ${output_path_pref}


#     for context_len in "${context_lens[@]}"
#     do
#         # echo $model_substr
#         python src/prompt_model.py --base_model $model --output_dir_path "${output_path_pref}"len_$context_len --prompt_context_len $context_len --model_type pretrained --data_split train
#     done
    
# done

# # lora-ft
# models=("EleutherAI/pythia-160m" "EleutherAI/pythia-410m" "EleutherAI/pythia-1.4b")
# output_path_prefix=("results/pythia-160m/lora-ft/epoch_20/train_set_completions/" "results/pythia-410m/lora-ft/epoch_20/train_set_completions/" "results/pythia-1.4b/lora-ft/epoch_20/train_set_completions/")
# peft_paths=("models/lora/pythia-160m/final_model" "models/lora/pythia-410m/final_model" "models/lora/pythia-1.4b/final_model" "models/lora/pythia-6.9b/final_model")

# for i in "${!models[@]}"
# do 
    
#     model=${models[i]}
#     output_path_pref=${output_path_prefix[i]}
#     peft_path=${peft_paths[i]}
#     # echo ${model} ${output_path_pref}


#     for context_len in "${context_lens[@]}"
#     do
#         echo python src/prompt_model.py --base_model $model --peft_weights_path $peft_path --output_dir_path "${output_path_pref}"len_$context_len--prompt_context_len $context_len --model_type lora  --data_split train
#         python src/prompt_model.py --base_model $model --peft_weights_path $peft_path --output_dir_path "${output_path_pref}"len_$context_len --prompt_context_len $context_len --model_type lora  --data_split train
#     done
    
# done

# # full-ft
# models=("models/full-ft/pythia-160m/final_model" "models/full-ft/pythia-410m/final_model")
# output_path_prefix=("results/pythia-160m/full-ft/epoch_20/train_set_completions/" "results/pythia-410m/full-ft/epoch_20/train_set_completions/")

models=("models/full-ft/pythia-1.4b/final_model")
output_path_prefix=("results/pythia-1.4b/full-ft/epoch_20/train_set_completions/")

for i in "${!models[@]}"
do 
    
    model=${models[i]}
    output_path_pref=${output_path_prefix[i]}
    # echo ${model} ${output_path_pref}


    for context_len in "${context_lens[@]}"
    do
        # echo $model_substr
        python src/prompt_model.py --base_model $model --output_dir_path "${output_path_pref}"len_$context_len --prompt_context_len $context_len --model_type full-ft  --data_split train
    done
    
done
