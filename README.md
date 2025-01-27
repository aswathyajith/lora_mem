## Memorization in LoRA fine-tuned models 

On DSI cluster, request a GPU with `srun -p clab,general -t 720:00 --gres=gpu:1 --mem 16G --pty /bin/bash` before running the steps below.

1. Create environment and `cd` to this directory: 
    `conda env create -f environment.yml`
    `conda activate lora_mem`

2. Fine-tune models (with lora or full param finetuning): 
    `./src/submit_script_finetuning.sh`

3. Generate model completions for each sample in the input
    `./src/submit_script_model_gens.sh`

4. Measure memorization in fine-tuned models: Prompt the models with the first k tokens of the prompt (k = 1, 2, 4, 8, 16).
    `./src/submit_script_partial_mem.sh`
    `./src/submit_script_exact_mem.sh`

4. Plot memorization across fine-tuning methods for each model 
    `./src/submit_script_plotting.sh`

To compute next token probabilities on a dataset: 
1. For each fine-tuning method, at a given checkpoint, compute the next token probabilities.
    `python src/next_token_probs.py` (if fine-tuning corpus)
    `python src/next_token_probs.py` (if pretraining corpus)

## Notes
We compare the two finetuning methods based on the probabilties they assign to the next token conditioned on tokens seen so far in the finetuning and pretraining corpus. We find that the two methods assign different probabilities to the actual next token, especially when the PMI between the current word and the next word is not high. In other words, when two consecutive tokens frequently appear together, the two methods do not differ much in probability, but when they don't, LoRA and full fine-tuning assign different probabilities.
