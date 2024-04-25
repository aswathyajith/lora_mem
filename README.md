## Memorization in LoRA fine-tuned models 

1. Create environment: 
    `conda env create -f environment.yml`

2. Fine-tune models (with lora or full param finetuning): 
    `python src/finetune_model.py --base_model EleutherAI/pythia-1.4b --model_config_path configs/model_configs/pythia-1.4b.json --sft_config_path configs/model_configs/sft_configs.json --lora_config_path configs/lora_config.json --lora`