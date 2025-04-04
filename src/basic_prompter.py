import torch 
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
from trl.trainer import ConstantLengthDataset
import random
import numpy as np
import pandas as pd
import os
from time import sleep
from collections import defaultdict

def load_model(model_name: str, 
               lora_adapter_path: str, 
               merge_and_unload: bool = False):
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

def preprocess_dataset(ds, dataset="bigbio/muchmore:muchmore_en_bigbio_kb"):
    print("dataset name: ", dataset)
    if dataset == "bigbio/muchmore:muchmore_en_bigbio_kb":
        def extract_text(row):
            row["text"] = row["passages"][0]["text"][0]
            return row
        
        ds = ds.map(extract_text)
    return ds

def decode_input_ids(examples: dict, 
                    tokenizer: AutoTokenizer, 
                    text_field: str = "text"):
    '''
    Decodes all input_ids in the batch at once
    '''
    # Decode all input_ids in the batch at once
    texts = [tokenizer.decode(ids) for ids in examples["input_ids"]]
    examples[text_field] = texts  
    masks = [[1] * len(input_id) for input_id in examples["input_ids"]]
    del examples["labels"]
    examples["attention_mask"] = masks
    return examples

def pack_dataset(dataset: Dataset, 
                tokenizer: AutoTokenizer, 
                max_length: int, 
                batch_size: int = 64,
                text_field: str = "text"):
    '''
    Packs dataset into a constant length dataset for inference.
    Each sample in the dataset is packed into a sequence of 
    length <max_length> and the eos token is appended to the end
    of each sequence.
    Returns a new dataset with the packed sequences.
    '''
    
    packed_ds = ConstantLengthDataset(
        tokenizer,
        dataset,  # Your original dataset
        dataset_text_field=text_field,  # The field containing your text
        seq_length=max_length,  # Desired sequence length
        infinite=False,  # Set to True if you want it to cycle indefinitely
        chars_per_token=3.6,  # Approximate number of characters per token
        shuffle=True,  # Whether to shuffle the dataset
        append_concat_token=True  # Whether to append a concatenation token between sequences
    )
    
    # Convert the ConstantLengthDataset to a Dataset and decode the input_ids
    new_ds = Dataset.from_generator(lambda: (yield from packed_ds))
    new_ds = new_ds.map(
        decode_input_ids,
        fn_kwargs={"tokenizer": tokenizer, "text_field": text_field},
        batched=True,
        batch_size=batch_size,  
    )
    return new_ds

def pack_dataset_copy(ds, tokenizer, max_length, text_field="text"):
    # Convert dataset to iterable dataset
    ds = ds.to_iterable_dataset()
    chunk_ids, chunk_masks, chunk_texts = [], [], []
    buffer_len = 10
    i = 0
    while True:
        buffer = []
        buffer_input_ids = []
        buffer_masks = []
        for i in range(buffer_len):
            example = ds.next()
            if example is None:
                break
            buffer.append(example)
            buffer_input_ids += example["input_ids"] + [tokenizer.eos_token_id]
            buffer_masks += example["attention_mask"] + [1]

        for i in range(0, len(buffer_input_ids), max_length):
            chunked_ids = buffer_input_ids[i:i+max_length]
            chunked_mask = buffer_masks[i:i+max_length]
            chunked_text = tokenizer.decode(chunked_ids)
            chunk_ids.append(chunked_ids)
            chunk_masks.append(chunked_mask)
            chunk_texts.append(chunked_text)
        break
    packed_ds = Dataset.from_dict({
        "input_ids": chunk_ids,
        "attention_mask": chunk_masks,
        text_field: chunk_texts
    })

    return packed_ds

def chunk_long_sequences(sample, tokenizer, max_length, text_field="text"):
    # Function to chunk sequences for packing

    chunk_ids = []
    chunk_masks = []
    chunk_texts = []

    for ids, mask in zip(sample["input_ids"], sample["attention_mask"]):
        for i in range(0, len(ids), max_length):
            chunked_ids = ids[i:i+max_length]
            chunked_mask = mask[i:i+max_length]

            # Decode chunk ids to get text
            chunked_text = tokenizer.decode(chunked_ids)

            
            chunk_ids.append(chunked_ids)
            chunk_masks.append(chunked_mask)
            chunk_texts.append(chunked_text)

    chunks_dict = {
        'input_ids': chunk_ids,
        'attention_mask': chunk_masks,
        text_field: chunk_texts
    }

    return chunks_dict

def select_max_tokens(
        dataset: Dataset, 
        tokenizer: AutoTokenizer, 
        text_field: str, 
        n_train_tkns: float
    ):
    
    '''
    Select samples from a dataset to reach a token budget of n_train_tkns.
    Will truncate samples that are too long to fit into the token budget.
    Returns dataset with selected samples. 
    '''

    max_tokens_left = n_train_tkns
    last_sample = {}

    # Randomly sample from dataset until we have n_samples x max_length number of tokens
    for i, sample in enumerate(dataset):
        
        sample_len = len(sample['input_ids']) 

        if sample_len >= max_tokens_left:
            # Cannot fit entire sample, getting first max_tokens_left tokens
            print(max_tokens_left)
            s = sample['input_ids'][:int(max_tokens_left)]
            last_sample['input_ids'] = [s]
            last_sample[text_field] = [tokenizer.decode(s)]

            break 
            
        max_tokens_left -= (sample_len + 1)
    
    # Construct dataset with first i samples and remaining chunks of last sample
    ds = dataset.select(range(i))
    if len(last_sample) > 0:
        ds = concatenate_datasets([ds, Dataset.from_dict(last_sample)])
    
    print(f"Token budget: {n_train_tkns}")
    print(f"Number of documents selected: {len(ds)}")
    print(f"Number of tokens in selected documents: {sum([1 + len(sample['input_ids']) for sample in ds]) - 1}")
    return ds

# _, tokenizer = load_model("EleutherAI/pythia-410m", None)
# load_data(tokenizer, dataset="wikitext:wikitext-2-raw-v1", split="train", num_train=10, pad_sequences=False, text_field="text", streaming=False, packing=True)
def load_data(
        tokenizer: AutoTokenizer, 
        dataset: str = "wikitext:wikitext-2-raw-v1", 
        split: str = "train", 
        num_train: int = -1, 
        max_length: int = 128, 
        pad_sequences: bool = False, 
        text_field: str = "text", 
        streaming: bool = False, 
        packing: bool = False, 
        inference: bool = False, 
        n_train_tkns: int | None = None, 
        stream_size: int = 10000
    ): 
    '''
    Loads a dataset from Huggingface, processes it and returns a dataset object.
    '''
    print("packing: ", packing)
    ds_name = dataset.split(":")
    print(ds_name)
    if len(ds_name) > 1: # has subset 
        ds = load_dataset(ds_name[0], ds_name[1], streaming=streaming, split=split)

    else: 
        ds = load_dataset(ds_name[0], streaming=streaming, split=split)
    
    if n_train_tkns is not None and not streaming:
        num_train = -1

    if streaming:
        print("streaming: ", streaming)
        if n_train_tkns is None:
            assert num_train != -1, "num_train cannot be -1 for streaming datasets"
            ds = ds.shuffle(seed=42, buffer_size=50000).take(num_train)
        
        if n_train_tkns is not None:
            assert stream_size is not None, "stream_size must be specified for streaming datasets when n_train_tkns is not None"
            ds = ds.shuffle(seed=42, buffer_size=50000).take(stream_size)
        
        # Convert iterable dataset to dataset
        def generator(iterable_ds):
            yield from iterable_ds

        ds = Dataset.from_generator(generator, gen_kwargs={"iterable_ds": ds})
        
    ds = preprocess_dataset(ds, dataset)

    print("num_train: ", num_train)
    # print(ds[text_field][0][:2048], flush=True)
    
    # Remove duplicates
    set_seed(42)
    uniq_text = sorted(list(set(ds[text_field])))
    df = pd.DataFrame({text_field: uniq_text})
    uniq_ds = Dataset.from_pandas(df)
    print("# Total Train samples: ", len(ds))
    print("# unique Train samples: ", len(uniq_ds))
    ds = uniq_ds.map(
        tokenize, 
        batched=True, 
        fn_kwargs={
            "tokenizer": tokenizer, 
            "field": text_field, 
            "max_length": max_length, 
            "packing": packing
        }
    )

    if pad_sequences:
        print(f"Padding sequences with length < {max_length}\nNumber of samples: {len(ds)}")
    
    elif not packing: # truncation
        ds = ds.filter(lambda sample: sample['input_ids'][max_length-1] != tokenizer.eos_token_id)
        print(f"Number of samples with length >= {max_length}: {len(ds)}")
    
    # setting seed for reproducibility 
    # (we don't need to set this as the same value as the arg seed)
    ds = ds.shuffle(seed=42) 
    if ((num_train > 0) and (len(ds) > num_train)) or (n_train_tkns is not None):
        
        if n_train_tkns is not None:
            print(f"selecting sequences to get {n_train_tkns} token budget after packing")
            ds = select_max_tokens(ds, tokenizer, text_field, n_train_tkns)
            if streaming:
                n_tkns = sum([1 + len(sample['input_ids']) for sample in ds]) - 1
                assert n_tkns == n_train_tkns, f"Number of tokens in selected documents (|T| = {n_tkns}) is less than token budget (|T_b| = {n_train_tkns})\nIncrease the num_train or decrease n_train_tkns"

            if split == "train":
                print(f"Number of sequences in epoch: {n_train_tkns / max_length}")
                steps_per_epoch = np.ceil(n_train_tkns / (max_length * 256)).astype(int)
                print(f"Number of steps in epoch: {steps_per_epoch}")
                print(f"Total training steps: {steps_per_epoch * 10}")
                sleep(10)
        else:
            print(f"selecting {num_train} random samples")
            print(f"WARNING: fix_total_tokens is not enabled, {len(ds)} documents will be selected from the dataset")
            ds = ds.select(range(num_train))
    # exit()
    if packing and inference: # Pack dataset for inference
        print("Packing dataset for inference..")
        print("Number of samples in dataset BEFORE packing: ", len(ds))
        print("Constructing packed dataset..")
        ds = pack_dataset(ds, tokenizer, max_length, text_field)
        print("Number of samples in dataset AFTER packing: ", len(ds))

    print("Number of documents selected: ", len(ds), flush=True)
    print("==================\nExample sequence\n==================\n", ds[0])
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

def tokenize(sample, tokenizer, field='text', max_length=128, packing=False):
    if packing:
        return tokenizer(
            sample[field], 
            padding=False, 
            truncation=False, 
            max_length=None, 
            return_tensors=None
        )
    else:
        return tokenizer(
            sample[field], 
            padding='max_length', 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )

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