import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from trl.trainer import ConstantLengthDataset
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import chain
import os

def preprocess_dataset(
        ds: Dataset, 
        dataset_name: str
    ):
    '''
    Preprocesses a dataset for training. Currently no preprocessing is done.
    '''
    print("Preprocessing dataset: ", dataset_name)
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

def pack_dataset(
        dataset: Dataset, 
        tokenizer: AutoTokenizer, 
        max_length: int, 
        batch_size: int = 64,
        text_field: str = "text"
    ):
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

def download_sample(
        dataset: str, 
        N: int = 10000, 
        split: str = "train"
    ):
    '''
    Downloads a sample of N=10000 from a split of a dataset in streaming mode.
    '''

    ds_name = dataset.split(":")
    if len(ds_name) > 1: # has subset 
        ds = load_dataset(ds_name[0], ds_name[1], streaming=True, split=split)

    else: 
        ds = load_dataset(ds_name[0], streaming=True, split=split)

    ds = ds.shuffle(seed=42, buffer_size=50000).take(N)
    ds = Dataset.from_generator(lambda: (yield from ds))
    return ds

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
        n_tkns: int | str | None = None, 
        downloaded: bool = False, 
        **kwargs
    ): 
    '''
    Loads a dataset from Huggingface, processes it and returns a dataset object.
    '''
    print("packing: ", packing)

    if downloaded: 
        ds_local_path = os.path.join(os.environ["HF_DATASETS_CACHE"], dataset)
        print("Loading dataset from local path: ", ds_local_path)
        ds = load_from_disk(ds_local_path)
        ds = ds[split]
            
    else: 
        ds_name = dataset.split(":")
        print(ds_name)
        
        if len(ds_name) > 1: # has subset 
            ds = load_dataset(ds_name[0], ds_name[1], split=split)

        else: 
            ds = load_dataset(ds_name[0], split=split)
    
    if n_tkns is not None:
        if isinstance(n_tkns, str):
            n_tkns = int(float(n_tkns))
            num_train = -1 # Will load all samples in data

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
    if ((num_train > 0) and (len(ds) > num_train)) or (n_tkns is not None):
        
        if n_tkns is not None:
            print(f"selecting sequences to get {n_tkns} token budget after packing")
            ds = select_max_tokens(ds, tokenizer, text_field, n_tkns)
            if streaming:
                n_tkns = sum([1 + len(sample['input_ids']) for sample in ds]) - 1
                assert n_tkns == n_tkns, f"Number of tokens in selected documents (|T| = {n_tkns}) is less than token budget (|T_b| = {n_tkns})\nDownload more samples or decrease n_tkns"

            if split == "train":
                print(f"Number of sequences in epoch: {n_tkns / max_length}")
                steps_per_epoch = np.ceil(n_tkns / (max_length * 256)).astype(int)
                print(f"Number of steps in epoch: {steps_per_epoch}")
                print(f"Total training steps: {steps_per_epoch * 10}")
        else:
            print(f"selecting {num_train} random samples")
            print(f"WARNING: n_tkns is not specified, {len(ds)} documents will be selected from the dataset")
            ds = ds.select(range(num_train))
            
    if packing and inference: # Pack dataset for inference
        print("Packing dataset for inference..")
        print("Number of samples in dataset BEFORE packing: ", len(ds))
        print("Constructing packed dataset..")
        ds = pack_dataset(ds, tokenizer, max_length, text_field=text_field)
        print("Number of samples in dataset AFTER packing: ", len(ds))

    print("Number of documents selected: ", len(ds), flush=True)
    print("==================\nExample sequence\n==================\n", ds[0])
    return ds

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

def set_seed(seed: int):
    '''
    Sets the seed for the random number generator.
    '''
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


def get_tkn_counts(dataset: Dataset):

    """
    Computes the token counts for a tokenized dataset. 'input_ids' must be present in the dataset.  
    """

    assert "input_ids" in dataset.column_names, "input_ids must be a feature in the dataset"
    input_ids = dataset["input_ids"]
    input_ids_flat = list(chain.from_iterable(input_ids))
    uniq_ids = set(input_ids_flat)
    print("# uniq tokens in dataset:", len(uniq_ids), flush=True)
    
    print("Computing token frequencies..", flush=True)
    token_freqs = Counter(input_ids_flat)

    # Convert to dataframe
    df = pd.DataFrame({
        "token": list(token_freqs.keys()), 
        "freq": list(token_freqs.values())
    })

    return df


def get_pair_counts(dataset: Dataset):

    """
    Counts pairs of tokens that appear in the same sequence/context. 
    Input dataset must have 'input_ids' as a feature and must not be packed or padded. 
    """
    
    assert "input_ids" in dataset.column_names, "input_ids must be a feature in the dataset"

    input_ids = dataset["input_ids"]
    pair_token_freqs = defaultdict(int)
    print("Computing token frequencies..", flush=True)
    
    for i in input_ids:
        for p_x in range(len(i)): 
            x = i[p_x]
            for p_y in range(p_x+1, len(i)): 
                y = i[p_y]
                pair_token_freqs[(x, y)] += 1

    df = pd.DataFrame({
        "pair": list(pair_token_freqs.keys()), 
        "freq": list(pair_token_freqs.values())
    })
    return df
    
    
