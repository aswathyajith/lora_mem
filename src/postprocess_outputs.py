import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import textwrap
import json 
import ast
import argparse
import re
from tqdm import tqdm
import os
from shutil import rmtree
from datasets import Dataset, DatasetDict, load_from_disk
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

def process_df(inp_path, save_path, prefix): 
    df = pd.read_csv(inp_path)
    df = df.apply(context_processing, axis=1)
    # Rename cols with same name to include prefix 
    # df = df.rename(columns={
    #     "curr_token_prob": f"{prefix}_prob", 
    #     "curr_token_rank": f"{prefix}_rank", 
    #     "top_k_pred_tokens": f"{prefix}_top_k_pred_tokens", 
    #     "top_k_pred_probs": f"{prefix}_top_k_pred_probs"
    # })
    df.to_csv(save_path, index=False)
    
    return df

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_merge_config_path", type=str, default="configs/data_merge_config.json")
    parser.add_argument("--output_dir", type=str, default="data/output_token_info/pythia-1.4b/legal/seed_3")

    args = parser.parse_args()
    data_merge_config_path = args.data_merge_config_path
    output_dir = args.output_dir

    with open(data_merge_config_path, "r") as f: 
        model_map = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    for model, model_info in model_map.items(): 
        dataset_path = os.path.join(output_dir, model+"_processed.csv")
        if os.path.exists(dataset_path): 
            print(f"Skipping {model} as it is already processed")
            continue
        path = model_info["path"]
        if "processed_path" not in model_info: 
            model_info["processed_path"] = path.replace(".csv", "_processed.csv")

        processed_path = model_info["processed_path"]
        if os.path.exists(processed_path): 
            df = pd.read_csv(processed_path)
        else: 
            df = process_df(path, processed_path, model)
        
        # Convert to HuggingFace Dataset custom split
        sel_cols = ["seq_id", "context_len", "in_tokens", "curr_token", "prev_token", "curr_token_freq", "prev_token_freq", "token_in_context", "uniq_ctxt_tkns_count", "curr_token_prob", "curr_token_rank", "top_k_pred_tokens", "top_k_pred_probs"]

        df = df[sel_cols]
        
        dataset_path = os.path.join(output_dir, model+"_processed.csv")
        df.to_csv(dataset_path, index=False)
        print(f"Saved {model} results to {dataset_path}")

    # merge_cols = ["seq_id", "context_len", "in_tokens", "curr_token", "prev_token", "curr_token_freq", "prev_token_freq", "token_in_context", "uniq_ctxt_tkns_count"]
    # combined_df = pd.merge(model_map["lora"]["df"], model_map["full"]["df"], on=merge_cols)
    # combined_df = pd.merge(combined_df, model_map["base"]["df"], on=merge_cols)
    # combined_df.to_csv(output_path, index=False)
    