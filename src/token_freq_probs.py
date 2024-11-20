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

def get_next_token_probs(model, input_ids, token_freqs, pair_token_freqs): 
    freq_prob_data = []
    
    for instance_num, input_id in enumerate(input_ids):
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

        # Normalize probabilities for visualization
        next_token_probs = np.array(next_token_probs)
        norm_probs = next_token_probs#(next_token_probs - next_token_probs.min()) / (next_token_probs.max() - next_token_probs.min())
        # print(norm_probs.shape)

        # Get curr and next tokens 
        tkn_ids = [inp_id.item() for inp_id in input_id[:, :-1][0]]
        tkns = tokenizer.convert_ids_to_tokens(tkn_ids)

        next_token_ids = [inp_id.item() for inp_id in input_id[:, 1:][0]]
        next_tkns = tokenizer.convert_ids_to_tokens(next_token_ids)

        # Get counts of curr and next tokens
        tkn_counts = [token_freqs[tkn_id] for tkn_id in tkn_ids]
        next_tkn_counts = [token_freqs[tkn_id] for tkn_id in next_token_ids]
        pair_counts = [pair_token_freqs[(tkn_id, nxt_tkn_id)] for tkn_id, nxt_tkn_id in zip(tkn_ids, next_token_ids)]

        freq_prob_data.extend(list(zip(zip(tkns, next_tkns), zip(tkn_ids, next_token_ids), zip(tkn_counts, next_tkn_counts, pair_counts, norm_probs))))
        break

    return freq_prob_data

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--lora_adapter_path", default="models/lora/pythia-1.4b/r_16/lr_2e-4/early_stopping/num_train_4096/bsize_128/checkpoint-300")
    parser.add_argument("--full_model_path", default="models/full-ft/pythia-1.4b/lr_2e-6/early_stopping/num_train_4096/bsize_128/checkpoint-200")
    parser.add_argument("--split", default="train")
    parser.add_argument("--lora_save_path", default="results/pythia-1.4b/lora/r_16/lr_2e-4/early_stopping/pretraining/tkn_freq_probs_final.csv")
    parser.add_argument("--full_save_path", default="results/pythia-1.4b/full-ft/lr_2e-6/early_stopping/pretraining/tkn_freq_probs_final.csv")
    parser.add_argument("--num_train", default=4096)
    parser.add_argument("--sample_frac", type=float, default=0.03)
    parser.add_argument("--pretraining_corpus", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    base_model = args.base_model
    lora_adapter_path = args.lora_adapter_path 
    full_model_path = args.full_model_path
    split = args.split
    save_path_lora = args.lora_save_path
    save_path_full = args.full_save_path
    num_train = args.num_train 
    pretraining_corpus = args.pretraining_corpus
    sample_frac = args.sample_frac

    #Load models 
    full_model, tokenizer = load_model(model_name=full_model_path, lora_adapter_path=None)
    lora_model, tokenizer = load_model(model_name=base_model, lora_adapter_path=lora_adapter_path)

    # Load dataset to compute next token probs over 
    if pretraining_corpus: 
        input_ids = load_pretraining_data(max_seq_length=128)
        random.seed(123)
        sample_size = int(sample_frac * input_ids.shape[0])
        input_ids_idx = random.sample(range(input_ids.shape[0]), sample_size)
        # Debugging
        input_ids = input_ids[input_ids_idx]
        print(len(input_ids))
        
    else:
        if split == "validation": 
            num_train = -1
        dataset = load_data(tokenizer, dataset="wikitext:wikitext-2-raw-v1", split=split, num_train=num_train, max_length=128, batch_size=None, pad_sequences=False)
        input_ids = dataset["input_ids"]
        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # Count token frequencies in dataset
    token_freqs = Counter(chain.from_iterable(input_ids))

    # Obtain frequencies of pairs of tokens in each sequence 
    pair_token_freqs = defaultdict(int)
    for i in input_ids:
        for p_x in range(len(i)): 
            x = i[p_x]
            for p_y in range(p_x+1, len(i)): 
                y = i[p_y]
                pair_token_freqs[(x, y)] += 1
    # pair_token_freqs = Counter(chain.from_iterable([[(x, y) for x in i for y in i[:]] for i in input_ids]))
    print(list(pair_token_freqs.keys())[:5])

    models = [lora_model, full_model]
    save_paths = [save_path_lora, save_path_full]

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.to(device)
    print(input_ids.shape)
    input_ids = torch.reshape(input_ids, (input_ids.shape[0], 1, input_ids.shape[1]))
    
    for i, model in enumerate(models):
        token_freq_probs = get_next_token_probs(model=model, input_ids=input_ids, token_freqs=token_freqs, pair_token_freqs=pair_token_freqs)
        df = pd.DataFrame({
            "prev_token": [tkn_freq_tuple[0][0] for tkn_freq_tuple in token_freq_probs], 
            "curr_token": [tkn_freq_tuple[0][1] for tkn_freq_tuple in token_freq_probs], 
            "prev_token_id": [tkn_freq_tuple[1][0] for tkn_freq_tuple in token_freq_probs],
            "curr_token_id": [tkn_freq_tuple[1][1] for tkn_freq_tuple in token_freq_probs], 
            "prev_token_freq": [tkn_freq_tuple[2][0] for tkn_freq_tuple in token_freq_probs], 
            "curr_token_freq": [tkn_freq_tuple[2][1] for tkn_freq_tuple in token_freq_probs], 
            "pair_token_freq": [tkn_freq_tuple[2][2] for tkn_freq_tuple in token_freq_probs], 
            "curr_token_prob": [tkn_freq_tuple[2][3] for tkn_freq_tuple in token_freq_probs]
        })


        n_pairs = len(pair_token_freqs)
        n_tokens = len(token_freqs)

        df["pmi"] = np.log(df.pair_token_freq) + 2*np.log(n_tokens) - (np.log(n_pairs) + np.log(df.curr_token_freq) + np.log((df.prev_token_freq))) 
        print(df.head())

        # Save data
        path_to_save = save_paths[i]
        if not os.path.exists(os.path.dirname(path_to_save)): 
            os.makedirs(os.path.dirname(path_to_save))


        df.to_csv(path_to_save, index=False)
    



    

