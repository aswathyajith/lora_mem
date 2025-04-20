import torch 
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import os
from trl import SFTTrainer

def load_model(
        model_name: str, 
        lora_adapter_path: str, 
        merge_and_unload: bool = False
    ):
    '''
    Loads a model from HF repo / path (model_name) and 
    merges with LoRA adapter if specified.
    If merge_and_unload is set to True, 
    the LoRA adapter is merged into the base model.
    Returns model and tokenizer
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

    if lora_adapter_path is not None:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        if merge_and_unload:
            model = model.merge_and_unload()

    return model, tokenizer

def prompt_model(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        batch: list[str], 
        prompt_context_len: int, 
        print_seqs: bool = False, 
        tokenized: bool=False
    ):
    '''
    Prompts a model with a batch of sequences.
    '''
    if not tokenized: 
        full_inp_ids = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda").input_ids # encode batch of input seq
    else: 
        full_inp_ids = torch.Tensor(batch["input_ids"]).to("cuda")
    # print(full_inp_ids)
    input_ids = full_inp_ids[:,:prompt_context_len] # get the input tokens that we want to prompt model with 
    inp_prompt = tokenizer.batch_decode(input_ids)
    model_comp_ids = model.generate(input_ids, max_length=128, early_stopping=True, do_sample=False)
    model_comp = tokenizer.batch_decode(model_comp_ids)

    if print_seqs:
        print("TRAIN SEQS:", batch)
        print("INPUT PROMPTS:", inp_prompt)
        print("INPUT PROMPTS + MODEL COMPLETIONS:", model_comp)
    return batch, inp_prompt, model_comp, full_inp_ids, model_comp_ids

def compute_loss(
        model_path_or_hf_name: str,
        lora_adapter_path: str | None = None,
        model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
        dataset: Dataset | None = None,
        text_field: str = "text", 
        max_seq_length: int = 64,
        packing: bool = True,
        base_model: bool = False
    ):
    """Computes loss of a model on a dataset using SFTTrainer"""

    if (model is None) or (tokenizer is None):
        model, tokenizer = load_model(model_path_or_hf_name, lora_adapter_path)
    
    # Get training args from model path or lora adapter path
    if not base_model:
        if lora_adapter_path is None:
            path_to_args = os.path.join(model_path_or_hf_name, "training_args.bin")
        else:
            path_to_args = os.path.join(lora_adapter_path, "training_args.bin")

        args = torch.load(path_to_args, weights_only=False)
        args.do_train = False
        args.do_eval = True
    else:
        args = None

    trainer = SFTTrainer(
        model=model,
        train_dataset=None,
        eval_dataset=dataset,
        dataset_text_field=text_field,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        eval_packing=packing,
        args=args
    )
    
    eval_results = trainer.evaluate()
    return eval_results