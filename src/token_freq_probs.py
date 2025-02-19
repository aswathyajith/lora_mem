import argparse 
from basic_prompter import * 
from collections import Counter, defaultdict
from itertools import chain, islice
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
from utils import load_pretraining_data
import random
from ast import literal_eval

def get_next_token_probs(model, input_ids, token_freqs, pair_token_freqs, top_k=10): 
    freq_prob_data = []
    # top_k_token_ids 
    device = input_ids.device

    for instance_num, input_id in enumerate(input_ids):
        next_token_ranks = []
        top_k_tokens = torch.tensor([], device=device, dtype=torch.int64)
        top_k_probs = torch.tensor([], device=device, dtype=torch.float32)
        print(instance_num)
        # Get probabilities of the actual next token at each position
        next_token_probs = []
        with torch.no_grad():
            outputs = model(input_id)
            logits = outputs.logits

            # Calculate probability for the actual next token at each position
            for i in range(logits.size(1) - 1):
                next_token_logits = logits[:, i, :]
                next_token_probs_dist = F.softmax(next_token_logits, dim=-1)
                actual_next_token_id = input_id[0, i + 1]
                actual_next_token_prob = next_token_probs_dist[0, actual_next_token_id].item()
                next_token_probs.append(actual_next_token_prob)

                # Calculate rank of the actual next token
                _, indices = torch.sort(next_token_probs_dist[0], descending=True)
                actual_next_token_rank = torch.where(indices == actual_next_token_id)[0][0].item() + 1
                next_token_ranks.append(actual_next_token_rank)

                # Get top k predictions
                top_k_token_preds = torch.topk(next_token_probs_dist[0], top_k)
                top_k_token_ids = top_k_token_preds.indices.reshape(1, -1)
                top_k_token_probs = torch.round(top_k_token_preds.values, decimals=3).reshape(1, -1)
                
                top_k_tokens = torch.cat((top_k_tokens, top_k_token_ids))
                top_k_probs = torch.cat((top_k_probs, top_k_token_probs))

        # Normalize probabilities for visualization
        next_token_probs = np.array(next_token_probs)
        next_token_ranks = np.array(next_token_ranks)
        top_k_tokens = top_k_tokens.cpu().numpy()
        top_k_probs = top_k_probs.cpu().numpy()

        norm_probs = next_token_probs#(next_token_probs - next_token_probs.min()) / (next_token_probs.max() - next_token_probs.min())
        # print(norm_probs.shape)

        # Get curr and next tokens 
        prev_token_ids = [inp_id.item() for inp_id in input_id[:, :-1][0]]
        prev_tokens = tokenizer.convert_ids_to_tokens(prev_token_ids)

        curr_token_ids = [inp_id.item() for inp_id in input_id[:, 1:][0]]
        curr_tokens = tokenizer.convert_ids_to_tokens(curr_token_ids)

        in_token_ids = [input_id[:, :-1][0][max(0, i-128):i+1] for i in range(len(input_id[:, :-1][0]))]
        in_tkns = tokenizer.batch_decode(in_token_ids)

        # Get counts of curr and next tokens
        tkn_counts = [token_freqs[tkn_id] for tkn_id in prev_token_ids]
        next_tkn_counts = [token_freqs[tkn_id] for tkn_id in curr_token_ids]
        pair_counts = [pair_token_freqs[(tkn_id, nxt_tkn_id)] for tkn_id, nxt_tkn_id in zip(prev_token_ids, curr_token_ids)]

        instance_nums = [instance_num] * len(prev_tokens)

        freq_prob_data.extend(list(
            zip(
                instance_nums, # sequence id
                zip(prev_tokens, curr_tokens, in_tkns), # tokens
                zip(prev_token_ids, curr_token_ids, in_token_ids), # token ids 
                zip(tkn_counts, next_tkn_counts, pair_counts, norm_probs), # counts and probabilities
                np.array(next_token_ranks), # rank of actual next token
                top_k_tokens, # top k predictions
                top_k_probs) # top k prediction probabilities
                )
            )

    return freq_prob_data

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--full_model_path", type=str, default=None, help="Path to the full-ft model") # models/full-ft/pythia-1.4b/lr_2e-6/early_stopping/num_train_4096/bsize_128/checkpoint-80
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Path to the lora adapter") # "models/lora/pythia-1.4b/r_16/lr_2e-4/early_stopping/num_train_4096/bsize_128/seed_1/final_model"
    
    parser.add_argument("--base_save_path", type=str, default=None, help="Path to save the base model generation outputs") # "results/pythia-1.4b/base_model/num_train_4096/tkn_freq_probs_base.csv"
    parser.add_argument("--lora_save_path" , type=str, default=None, help="Path to save the lora model generation outputs") # "results/pythia-1.4b/lora/r_16/lr_2e-4/early_stopping/num_train_4096/bsize_128/seed_1/tkn_freq_probs_best.csv"
    parser.add_argument("--full_save_path", type=str, default=None, help="Path to save the full-ft model generation outputs") # "results/pythia-1.4b/full-ft/lr_2e-6/early_stopping/num_train_4096/bsize_128/tkn_freq_probs_best.csv"
    
    parser.add_argument("--dataset", default="wikitext:wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_train", type=int, default=1024, help="Number of examples to evaluate on in the finetuning train corpus")
    parser.add_argument("--shuffle_train", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pretrain_sample_frac", type=float, default=0.001)
    parser.add_argument("--freq_save_dir", type=str, default="results/token_freqs/legal/train/sample_1024")
    parser.add_argument("--pretraining_corpus", action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, help="Skips generation if the save path already exists")

    args = parser.parse_args()
    base_model_name = args.base_model
    lora_adapter_path = args.lora_adapter_path 
    full_model_path = args.full_model_path
    dataset = args.dataset
    split = args.split
    save_path_base = args.base_save_path
    save_path_lora = args.lora_save_path
    save_path_full = args.full_save_path
    num_train = args.num_train 
    pretraining_corpus = args.pretraining_corpus
    sample_frac = args.pretrain_sample_frac
    max_length = args.max_length
    shuffle = args.shuffle_train
    skip_existing = args.skip_existing
    freq_save_dir = args.freq_save_dir
    seed = 123 # Seed for shuffling the dataset
    models = []
    save_paths = []

    tokenizer = None

    #Load models 
    if (save_path_base is not None): 
        if skip_existing and os.path.exists(save_path_base): 
            print(f"Skipping base model {base_model_name} as it already exists at {save_path_base}")
        else: 
            base_model, tokenizer = load_model(model_name=base_model_name, lora_adapter_path=None)
            models.append(base_model)
            save_paths.append(save_path_base)

    if (save_path_lora is not None):
        if skip_existing and os.path.exists(save_path_lora): 
            print(f"Skipping lora model '{lora_adapter_path}' generations as it already exists at {save_path_lora}")
        else: 
            lora_model, tokenizer = load_model(model_name=base_model_name, lora_adapter_path=lora_adapter_path)
            models.append(lora_model)
            save_paths.append(save_path_lora)

    if (save_path_full is not None):
        if skip_existing and os.path.exists(save_path_full):
            print(f"Skipping full model '{full_model_path}' generations as it already exists at {save_path_full}")
        else: 
            full_model, tokenizer = load_model(model_name=full_model_path, lora_adapter_path=None)
            models.append(full_model)
            save_paths.append(save_path_full)

    if tokenizer is None: 
        print("No model provided, loading base model for tokenizer..")
        _, tokenizer = load_model(model_name=base_model_name, lora_adapter_path=None)

    # Load dataset to compute next token probs over 
    if pretraining_corpus: 
        input_ids = load_pretraining_data(max_seq_length=max_length)
        random.seed(seed)
        sample_size = int(sample_frac * input_ids.shape[0])
        input_ids_idx = random.sample(range(input_ids.shape[0]), sample_size)
        # Debugging
        input_ids = input_ids[input_ids_idx]
        print(len(input_ids))
        
    else:
        if split == "validation": 
            num_train = -1
        dataset = load_data(tokenizer, dataset=dataset, split=split, num_train=num_train, max_length=max_length, batch_size=None, pad_sequences=False, shuffle=shuffle, seed=seed)
        input_ids = dataset["input_ids"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(freq_save_dir, exist_ok=True)

    # Count token frequencies in dataset
    tkn_freq_path = os.path.join(freq_save_dir, "tkn_freq.csv")
    if not os.path.exists(tkn_freq_path): 
        token_freqs = Counter(chain.from_iterable(input_ids))
        # Convert to dataframe
        df = pd.DataFrame({"token": list(token_freqs.keys()), "freq": list(token_freqs.values())})
        df.to_csv(tkn_freq_path, index=False)
    else: 
        df = pd.read_csv(tkn_freq_path)
        token_freqs = dict(zip(df["token"], df["freq"]))

    # Obtain frequencies of pairs of tokens in each sequence 
    
    pair_freq_path = os.path.join(freq_save_dir, "pair_freq.csv")
    if not os.path.exists(pair_freq_path): 
        pair_token_freqs = defaultdict(int)
        for i in input_ids:
            for p_x in range(len(i)): 
                x = i[p_x]
                for p_y in range(p_x+1, len(i)): 
                    y = i[p_y]
                    pair_token_freqs[(x, y)] += 1

        # Convert to dataframe
        df = pd.DataFrame({"pair": list(pair_token_freqs.keys()), "freq": list(pair_token_freqs.values())})
        df.to_csv(pair_freq_path, index=False)
    else: 
        df = pd.read_csv(pair_freq_path)
        # convert pair column type from str to tuple
        
        df["pair"] = df["pair"].apply(literal_eval)
        pair_token_freqs = dict(zip(df["pair"], df["freq"]))

    # pair_token_freqs = Counter(chain.from_iterable([[(x, y) for x in i for y in i[:]] for i in input_ids]))
    # print(list(pair_token_freqs.keys())[:5])

    model_path_pairs = list(zip(models, save_paths))

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.to(device)
    input_ids = torch.reshape(input_ids, (input_ids.shape[0], 1, input_ids.shape[1]))
    
    for model, save_path in model_path_pairs:
        token_freq_probs = get_next_token_probs(model=model, input_ids=input_ids, token_freqs=token_freqs, pair_token_freqs=pair_token_freqs)
        df = pd.DataFrame({
            "seq_id": [tkn_freq_tuple[0] for tkn_freq_tuple in token_freq_probs],
            "prev_token": [tkn_freq_tuple[1][0] for tkn_freq_tuple in token_freq_probs], 
            "curr_token": [tkn_freq_tuple[1][1] for tkn_freq_tuple in token_freq_probs], 
            "prev_token_id": [tkn_freq_tuple[2][0] for tkn_freq_tuple in token_freq_probs],
            "curr_token_id": [tkn_freq_tuple[2][1] for tkn_freq_tuple in token_freq_probs], 
            "in_tokens": [tkn_freq_tuple[1][2] for tkn_freq_tuple in token_freq_probs], 
            "in_token_ids": [tkn_freq_tuple[2][2].cpu() for tkn_freq_tuple in token_freq_probs],
            "prev_token_freq": [tkn_freq_tuple[3][0] for tkn_freq_tuple in token_freq_probs], 
            "curr_token_freq": [tkn_freq_tuple[3][1] for tkn_freq_tuple in token_freq_probs], 
            "pair_token_freq": [tkn_freq_tuple[3][2] for tkn_freq_tuple in token_freq_probs], 
            "curr_token_prob": [tkn_freq_tuple[3][3] for tkn_freq_tuple in token_freq_probs], 
            "curr_token_rank": [tkn_freq_tuple[4] for tkn_freq_tuple in token_freq_probs], 
            "top_k_pred_tokens": [tkn_freq_tuple[5] for tkn_freq_tuple in token_freq_probs], 
            "top_k_pred_probs": [tkn_freq_tuple[6] for tkn_freq_tuple in token_freq_probs]
        })


        n_pairs = len(pair_token_freqs)
        n_tokens = len(token_freqs)

        df["pmi"] = np.log(df.pair_token_freq) + 2*np.log(n_tokens) - (np.log(n_pairs) + np.log(df.curr_token_freq) + np.log((df.prev_token_freq))) 
        print(df[["curr_token", "curr_token_rank", "top_k_pred_tokens", "top_k_pred_probs"]].head())

        # Save data
        if not os.path.exists(os.path.dirname(save_path)): 
            os.makedirs(os.path.dirname(save_path))

        df.to_csv(save_path, index=False)
    



    

