## How Memorization Paves the Path for Generalization during Finetuning

### Abstract
Large Language Models (LLMs) pre-trained on generic corpora can be adapted to specific domains and tasks by fine-tuning their weights. Due to compute and memory limitations, parameter-efficient methods like LoRA have become a popular alternative to full fine-tuning of model parameters (i.e. updating a few parameters instead of all model parameters). While full fine-tuning and LoRA fine-tuning can achieve similar validation loss on fine-tuning data distributions, there has been limited work exploring differences between their generative behaviors. In this work, we find that even when full finetuning and LoRA finetuning exhibit similar performance on the validation set, full finetuning "memorizes" more of the data it was finetuned on than LoRA. Previous work has shown that full finetuning can generalize more to out-of-distribution data than LoRA. While it is generally believed that memorization of training data leads to poor generalizability on out of domain distributions, we find empirical evidence that training data memorization during finetuning is crucial for generalization to out-of-distribution data.

On DSI cluster, request a GPU with `srun -p clab,general -t 720:00 --gres=gpu:1 --mem 16G --pty /bin/bash` before running the steps below.

### Output Visualization: 

To run the streamlit app comparing the outputs of LoRA and full fine-tuning, make sure `pandas` and `streamlit` are installed.

    `pip install streamlit pandas`

Run the streamlit server from the project root dir: 
    `streamlit run demo/app.py --server.port 8501`

[OPTIONAL] Set up port forwarding on local machine (if server is running on remote machine):

    `ssh -L 8501:localhost:8501 user@remote`

### Steps to Reproduce Results

1. Create environment and `cd` to this directory: 
    `conda env create -f environment.yml`
    `conda activate lora_mem`

2. Fine-tune models (with lora or full param finetuning): 
    `./src/scripts/<ft_method>/submit_script_finetuning.sh`

    To check if finetunde models are ready, run `python src/check_finetuning.py` with the appropriate arguments.

3. Hyperparameter search (find the best learning rate for the finetuned models): 
    `python src/find_optimal_lr.py`

4. Compute next token probabilities, actual token position, top k predictions, and token frequencies: 
    `./src/scripts/tkn_freq_probs.sh`

5. Generate postprocess config: 
    `python src/generate_postprocess_config.py`

6. Postprocess outputs of models: 
    `./src/scripts/postprocess_outputs.sh`