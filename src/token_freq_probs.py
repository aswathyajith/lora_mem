import argparse 
from basic_prompter import * 
from collections import Counter
from itertools import chain
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
from utils import load_pretraining_data
import random
import json
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
        next_tkn_counts = [token_freqs[tkn_id] for tkn_id in curr_token_ids]
        tkn_counts = [token_freqs[tkn_id] for tkn_id in prev_token_ids]
        
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
    parser.add_argument("--base_model", required=True)
    # parser.add_argument("--full_model_path", type=str, default=None, help="Path to the full-ft model") # models/full-ft/pythia-1.4b/lr_2e-6/early_stopping/num_train_4096/bsize_128/checkpoint-80
    # parser.add_argument("--lora_adapter_path", type=str, default=None, help="Path to the lora adapter") # "models/lora/pythia-1.4b/r_16/lr_2e-4/early_stopping/num_train_4096/bsize_128/seed_1/final_model"
    
    # parser.add_argument("--base_save_path", type=str, default=None, help="Path to save the base model generation outputs") # "results/pythia-1.4b/base_model/num_train_4096/tkn_freq_probs_base.csv"
    # parser.add_argument("--lora_save_path" , type=str, default=None, help="Path to save the lora model generation outputs") # "results/pythia-1.4b/lora/r_16/lr_2e-4/early_stopping/num_train_4096/bsize_128/seed_1/tkn_freq_probs_best.csv"
    # parser.add_argument("--full_save_path", type=str, default=None, help="Path to save the full-ft model generation outputs") # "results/pythia-1.4b/full-ft/lr_2e-6/early_stopping/num_train_4096/bsize_128/tkn_freq_probs_best.csv"
    
    parser.add_argument("--model_data_gen_config", type=str, default="configs/tkn_freq_prob_config.json")
    parser.add_argument("--domain", default="legal")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pretrain_corpus_max_length", type=int, default=2048)
    parser.add_argument("--pretrain_corpus_sample_frac", type=float, default=0.001)
    parser.add_argument("--freq_save_dir", type=str, default="results/token_freqs")
    parser.add_argument("--model_dir", type=str, default="models/pythia-1.4b")
    parser.add_argument("--model_outputs_dir", type=str, default="results/model_gens/pythia-1.4b")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, help="Skips generation if the path already exists")

    args = parser.parse_args()
    model_data_gen_config = args.model_data_gen_config
    domain = args.domain
    split = args.split
    model_dir = args.model_dir
    model_outputs_dir = args.model_outputs_dir
    base_model_name = args.base_model
    max_length = args.pretrain_corpus_max_length
    sample_frac = args.pretrain_corpus_sample_frac
    skip_existing = args.skip_existing
    freq_save_dir = args.freq_save_dir
    seed = args.seed # Seed for identifying finetuning run
    models = []
    save_paths = []

    # Load model data gen config
    _, tokenizer = load_model(model_name=base_model_name, lora_adapter_path=None)

    # Load dataset to compute next token probs over 
    if split == "pretrain": 
        freq_save_dir = os.path.join(freq_save_dir, "pretrain")
        tkn_freq_path = os.path.join(freq_save_dir, "tkn_freq.csv")
        pair_freq_path = os.path.join(freq_save_dir, "pair_freq.csv")

        if not (os.path.exists(tkn_freq_path) or os.path.exists(pair_freq_path)): 
            input_ids = load_pretraining_data(max_seq_length=max_length)
            random.seed(seed)
            sample_size = int(sample_frac * input_ids.shape[0])
            input_ids_idx = random.sample(range(input_ids.shape[0]), sample_size)
            input_ids = input_ids[input_ids_idx]
            print("Pretraining sample size: ", len(input_ids))

        if os.path.exists(tkn_freq_path): # Load from disk if path exists
            df = pd.read_csv(tkn_freq_path)
            token_freqs = dict(zip(df["token"], df["freq"]))
        else: # Compute and save to disk if path does not exist
            token_freqs = Counter(chain.from_iterable(input_ids))
            df = pd.DataFrame({"token": list(token_freqs.keys()), "freq": list(token_freqs.values())})
            df.to_csv(tkn_freq_path, index=False)
            

        if os.path.exists(pair_freq_path): # Load from disk if path exists 
            df = pd.read_csv(pair_freq_path)
            # convert pair column type from str to tuple
            df["pair"] = df["pair"].apply(literal_eval)
            pair_token_freqs = dict(zip(df["pair"], df["freq"]))
        else: # Compute and save to disk if path does not exist
            pair_token_freqs = compute_pair_freqs(input_ids)
            df = pd.DataFrame({"pair": list(pair_token_freqs.keys()), "freq": list(pair_token_freqs.values())})
            df.to_csv(pair_freq_path, index=False)

        exit()

    with open(model_data_gen_config, "r") as f:
        model_data_gen_config = json.load(f)

    domain_config = model_data_gen_config[domain]

    for dataset, dataset_config in domain_config.items():
        if "[skip]" in dataset: # Skipping dataset
            continue

        print(f"generating token freq and token probs for {dataset}")

        freq_save_dir = os.path.join(freq_save_dir, dataset_config["subdir"])
        model_dir = os.path.join(model_dir, dataset_config["subdir"])
        print("model_dir: ", model_dir)
        model_outputs_dir = os.path.join(model_outputs_dir, dataset_config["subdir"])
        n_samples = dataset_config["n_samples"]
        sample_dir = f"sample_{n_samples}" if n_samples != -1 else "sample_all" 
        print("Models: ", dataset_config["models"], flush=True)
        for model in dataset_config["models"]:
            # model_path = os.path.join(model_dir, domain, dataset)
            print("model: ", model)
            if model == "base": 

                save_path_base = os.path.join(model_outputs_dir, split, sample_dir, "base", f"tkn_freq_probs.csv")
                if skip_existing and os.path.exists(save_path_base): 
                    print(f"Skipping base model {base_model_name} as it already exists at {save_path_base}")
                    continue

                base_model, tokenizer = load_model(base_model_name, None)
                # base_model = "base_model" # Debugging
                models.append(base_model)
                save_paths.append(save_path_base)
                
            elif model == "lora": 
                print("lora", flush=True)
                for rank, lora_lr in dataset_config["lora_rank_lr"].items():
                    lora_model_path = os.path.join(model_dir, "lora", f"r_{rank}", f"lr_{lora_lr}", f"seed_{seed}", "final_model")
                    
                    save_path_lora = os.path.join(model_outputs_dir, split, sample_dir, "lora", f"r_{rank}", f"lr_{lora_lr}", f"seed_{seed}", f"tkn_freq_probs.csv")
                    print("save_path_lora: ", skip_existing and os.path.exists(save_path_lora))
                    if skip_existing and os.path.exists(save_path_lora): 
                        print(f"Skipping lora model {lora_model_path} as it already exists at {save_path_lora}", flush=True)
                        continue

                    lora_model = None
                    if os.path.exists(lora_model_path): 
                        # lora_model = "lora_model" # Debugging
                        lora_model, tokenizer = load_model(base_model_name, lora_model_path)
                        print("lora_model: ", lora_model)
                    models.append(lora_model)
                    save_paths.append(save_path_lora)

            elif model == "full": 
                full_lr = dataset_config["full_lr"]
                full_model_path = os.path.join(model_dir, "full-ft", f"lr_{full_lr}", f"seed_{seed}", "final_model")
                
                save_path_full = os.path.join(model_outputs_dir, split, sample_dir, "full-ft", f"lr_{full_lr}", f"seed_{seed}", f"tkn_freq_probs.csv")
                # if skip_existing and os.path.exists(save_path_full): 
                #     print(f"Skipping full model {full_model_path} as it already exists at {save_path_full}")
                #     continue

                full_model = None
                
                if os.path.exists(full_model_path): 
                    print("Loading full model from ", full_model_path)
                    # full_model = "full_model" # Debugging
                    full_model, tokenizer = load_model(full_model_path, None)
                else:
                    print(f"Full model {full_model_path} does not exist")
                print("full_model: ", full_model)
                
                models.append(full_model)
                save_paths.append(save_path_full)
            # print(save_paths)
    
        n_samples = dataset_config["n_samples"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(freq_save_dir, exist_ok=True)
        input_ids = None

        print(f"Getting token and pair frequencies for {dataset}..")
        tkn_freq_path = os.path.join(freq_save_dir, "tkn_freq_new.csv")
        pair_freq_path = os.path.join(freq_save_dir, "pair_freq_new.csv")

        if (not os.path.exists(tkn_freq_path)) or (not os.path.exists(pair_freq_path)): 
            data = load_data(tokenizer, dataset=dataset, split=split, num_train=n_samples, max_length=max_length, batch_size=None, pad_sequences=False)
            input_ids = data["input_ids"]
            
        if not os.path.exists(tkn_freq_path): 
            
            input_ids_flat = list(chain.from_iterable(input_ids))
            uniq_ids = set(input_ids_flat)
            print("# uniq ids:", len(uniq_ids), flush=True)
            print("Computing token frequencies..", flush=True)

            token_freqs = Counter(input_ids_flat)
            token_freqs_keys = set(token_freqs.keys())
            print("token_freqs: ", len(token_freqs_keys), flush=True)
            equal = uniq_ids == set(token_freqs.keys())
            print("uniq_ids == token_freqs.keys(): ", equal, flush=True)
            if not equal: 
                print("uniq_ids not in token_freqs keys: ", uniq_ids - token_freqs_keys, flush=True)
            # Convert to dataframe
            df = pd.DataFrame({"token": list(token_freqs.keys()), "freq": list(token_freqs.values())})
            df.to_csv(tkn_freq_path, index=False)
        else: 
            df = pd.read_csv(tkn_freq_path)
            token_freqs = dict(zip(df["token"], df["freq"]))
            token_freqs_keys = set(token_freqs.keys())
        
        if not os.path.exists(pair_freq_path):
            print("Computing pair frequencies..")
            pair_token_freqs = compute_pair_freqs(input_ids)
            df = pd.DataFrame({"pair": list(pair_token_freqs.keys()), "freq": list(pair_token_freqs.values())})
            df.to_csv(pair_freq_path, index=False)
        else: 
            df = pd.read_csv(pair_freq_path)
            df["pair"] = df["pair"].apply(literal_eval)
            pair_token_freqs = dict(zip(df["pair"], df["freq"]))

        model_path_pairs = list(zip(models, save_paths))
        
        if len(model_path_pairs) == 0: 
            print("No models to process")
            exit()

        if input_ids is None:
            data = load_data(tokenizer, dataset=dataset, split=split, num_train=n_samples, max_length=max_length, batch_size=None, pad_sequences=False)
            input_ids = data["input_ids"]
            
        input_ids_flat = list(chain.from_iterable(input_ids))
        uniq_ids = set(input_ids_flat)
        print("# uniq_ids: ", len(uniq_ids), flush=True)
        print("# token_freqs_keys: ", len(token_freqs_keys), flush=True)
        equal = uniq_ids == token_freqs_keys
        print("uniq_ids == token_freqs_keys: ", equal, flush=True)
        if not equal: 
            print("uniq_ids not in token_freqs_keys: ", uniq_ids - token_freqs_keys, flush=True)

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
    



    

