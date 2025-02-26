import pandas as pd
import json 
import ast
import argparse
import re
from tqdm import tqdm
import os
tqdm.pandas()


# Function to extract context length from df 
def context_processing(x):
    context_ids, token = x["in_token_ids"], x["curr_token_id"]
    context = ast.literal_eval(re.split(r'tensor\(|\].*', context_ids)[1] + ']')
    x["context_len"] = len(context)
    x["token_in_context"] = int(token in context)
    x["uniq_ctxt_tkns_count"] = len(set(context))
    return x
    
def get_context(x):
    return ast.literal_eval(re.split(r'tensor\(|\].*', x)[1] + ']')

def process_df(inp_path, save_path): 
    df = pd.read_csv(inp_path)
    df = df.apply(context_processing, axis=1)
    df.to_csv(save_path, index=False)
    
    return df

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_merge_config_path", type=str, default="configs/merge_config.json")
    parser.add_argument("--output_dir", type=str, default="data/output_token_info/pythia-1.4b")
    parser.add_argument("--data_dir", type=str, default="legal/cuad/train/sample_all")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    data_merge_config_path = args.data_merge_config_path
    data_dir = args.data_dir
    seed = args.seed
    output_data_dir = "/".join(data_dir.split("/")[:2])
    output_dir = os.path.join(args.output_dir, output_data_dir, f"seed_{args.seed}")
    

    with open(data_merge_config_path, "r") as f: 
        model_map = json.load(f)

    dataset_model = model_map[data_dir]

    os.makedirs(output_dir, exist_ok=True)
    for model, model_info in dataset_model.items(): 
        dataset_path = os.path.join(output_dir, model+"_processed.csv")
        if os.path.exists(dataset_path): 
            print(f"Skipping {model} as it is already processed")
            continue

        path = model_info["path"]
        if model != "base": 
            path = os.path.join(path, f"seed_{seed}", "tkn_freq_probs.csv")
        if "processed_path" not in model_info: 
            model_info["processed_path"] = path.replace(".csv", "_processed.csv")

        processed_path = model_info["processed_path"]
        if os.path.exists(processed_path): 
            df = pd.read_csv(processed_path)
        else: 
            df = process_df(path, processed_path)
        
        # Merge with pretraining data 
        pretraining_data = pd.read_csv("results/token_freqs/pretraining_corpus/tkn_freq.csv")
        pretraining_data.columns = ["curr_token_id", "pt_curr_token_freq"]
        df = pd.merge(df, pretraining_data, on="curr_token_id", how="left")
        df["pt_curr_token_freq"] = df["pt_curr_token_freq"].fillna(0) # impute missing values with 0 
        # Convert to HuggingFace Dataset custom split

        # Add relative prevalence 
        ft_tot_count = df["curr_token_freq"].sum()
        pt_tot_count = df["pt_curr_token_freq"].sum()
        df["rel_prev"] = (df["curr_token_freq"] / ft_tot_count) - df["pt_curr_token_freq"] / pt_tot_count

        sel_cols = ["seq_id", "context_len", "in_tokens", "curr_token", "prev_token", "curr_token_freq", "pt_curr_token_freq", "prev_token_freq", "token_in_context", "uniq_ctxt_tkns_count", "curr_token_prob", "curr_token_rank", "top_k_pred_tokens", "top_k_pred_probs", "rel_prev"]

        df = df[sel_cols]
        
        dataset_path = os.path.join(output_dir, model+"_processed.csv")
        df.to_csv(dataset_path, index=False)
        print(df.columns)

    # merge_cols = ["seq_id", "context_len", "in_tokens", "curr_token", "prev_token", "curr_token_freq", "prev_token_freq", "token_in_context", "uniq_ctxt_tkns_count"]
    # combined_df = pd.merge(model_map["lora"]["df"], model_map["full"]["df"], on=merge_cols)
    # combined_df = pd.merge(combined_df, model_map["base"]["df"], on=merge_cols)
    # combined_df.to_csv(output_path, index=False)
    