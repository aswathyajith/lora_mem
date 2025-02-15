## Memorization in LoRA fine-tuned models 

On DSI cluster, request a GPU with `srun -p clab,general -t 720:00 --gres=gpu:1 --mem 16G --pty /bin/bash` before running the steps below.

1. Create environment and `cd` to this directory: 
    `conda env create -f environment.yml`
    `conda activate lora_mem`

2. Fine-tune models (with lora or full param finetuning): 
    `./src/scripts/<ft_method>/submit_script_finetuning.sh`

3. Compute next token probabilities, actual token position, top k predictions, and token frequencies: 
    `./src/scripts/tkn_freq_probs.sh`

4. Postprocess outputs of models: 
    `./src/scripts/postprocess_outputs.sh`

## Notes
We compare the two finetuning methods based on the probabilties they assign to the next token conditioned on tokens seen so far in the finetuning and pretraining corpus. We find that the two methods assign different probabilities to the actual next token, especially when the PMI between the current word and the next word is not high. In other words, when two consecutive tokens frequently appear together, the two methods do not differ much in probability, but when they don't, LoRA and full fine-tuning assign different probabilities.
