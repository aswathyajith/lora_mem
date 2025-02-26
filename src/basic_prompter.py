import torch 
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# def load_model(model_name, lora_adapter_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             low_cpu_mem_usage=True,
#             return_dict=True,
#             torch_dtype=torch.float16,
#             device_map="auto",
#         )

#     if lora_adapter_path is not None:
#         model = PeftModel.from_pretrained(model, lora_adapter_path)
#         model = model.merge_and_unload()

#     return model, tokenizer

def load_model(model_name, lora_adapter_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # low_cpu_mem_usage=True,
                # return_dict=True,
                # torch_dtype=torch.float16,
                device_map="auto",
            )

        if lora_adapter_path is not None:
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            # model = model.merge_and_unload()

        return model, tokenizer

def preprocess_dataset(ds, dataset="bigbio/muchmore:muchmore_en_bigbio_kb"):
    print("dataset name: ", dataset)
    if dataset == "bigbio/muchmore:muchmore_en_bigbio_kb":
        def extract_text(row):
            row["text"] = row["passages"][0]["text"][0]
            return row
        
        ds = ds.map(extract_text)
    return ds
def load_data(tokenizer, dataset="wikitext:wikitext-2-raw-v1", split="train", num_train=-1, max_length=128, batch_size=None, pad_sequences=False, text_field="text", streaming=False): 
    
    ds_name = dataset.split(":")
    print(ds_name)
    if len(ds_name) > 1: # has subset 
        ds = load_dataset(ds_name[0], ds_name[1], streaming=streaming, split=split)

    else: 
        ds = load_dataset(ds_name[0], streaming=streaming, split=split)
    
    if streaming:
        ds = ds.shuffle(seed=42, buffer_size=20000).take(num_train)
        
        # Convert iterable dataset to dataset
        def generator(iterable_ds):
            yield from iterable_ds

        ds = Dataset.from_generator(generator, gen_kwargs={"iterable_ds": ds})
    ds = preprocess_dataset(ds, dataset)

    if dataset == "Rowan/hellaswag": 
         ds = ds.filter(lambda row: row['label'] != "") # Filter out rows without label 
         def combine_context_ending(row): 
            context = row['ctx']
            endings = row['endings']
            ending_corr_id = int(row['label'])
            ending_corr = endings[ending_corr_id]

            row["text"] = context + ' ' + ending_corr
            return row
         
         ds = ds.map(combine_context_ending)

    print("num_train: ", num_train)
    # print(ds[text_field][0][:2048], flush=True)
    
    # Remove duplicates
    uniq_text = sorted(list(set(ds[text_field])))
    df = pd.DataFrame({text_field: uniq_text})
    uniq_ds = Dataset.from_pandas(df)
    print("# unique Train samples: ", len(ds))
    ds = uniq_ds.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "field": text_field, "max_length": max_length})

    if not pad_sequences:
        ds = ds.filter(lambda sample: sample['input_ids'][max_length-1] != tokenizer.eos_token_id)

    print(f"Number of samples with length >= {max_length}: {len(ds)}")
        
    # setting seed for reproducibility 
    # (we don't need to set this as the same value as the arg seed)
    ds = ds.shuffle(seed=42) 
    if num_train > 0:
        ds = ds.select(range(num_train))
    
    print("Loaded dataset size: ", len(ds), flush=True)
    print(ds)
    return ds

def prompt_model(model, tokenizer, batch, prompt_context_len, print_seqs=False, tokenized=False):

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

def tokenize(sample, tokenizer, field='text', max_length=128):
    return tokenizer(sample[field], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to model or Huggingface model name", default="EleutherAI/pythia-410m")
    parser.add_argument("--lora_adapter_path", help="Path to fine-tuned LoRA adapter", default=None)
    parser.add_argument("--context_len", help="Input context length", default=None)
    args = parser.parse_args()
    model_path = args.model_path
    lora_adapter_path = args.lora_adapter_path

    model, tokenizer = load_model(model_path, lora_adapter_path)
    ds = load_data(tokenizer)["text"]
    prompt_model(model, tokenizer, ds, 32, True, False)

def get_comp(batch, model_path="models/full-ft/pythia-160m/num_train_1/checkpoint-1000", lora_adapter_path=None):
    model, tokenizer = load_model(model_path, lora_adapter_path)
    out = prompt_model(model, tokenizer, batch, 25, True, False)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")

def compute_pair_freqs(input_ids):
        pair_token_freqs = defaultdict(int)
        for i in input_ids:
            for p_x in range(len(i)): 
                x = i[p_x]
                for p_y in range(p_x+1, len(i)): 
                    y = i[p_y]
                    pair_token_freqs[(x, y)] += 1
        return pair_token_freqs