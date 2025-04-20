import pandas as pd
import argparse
import os
import numpy as np
import textwrap

def prep_df_for_merge(df, col_prefix, common_cols, merge_cols):
    df = df.rename(columns={"norm_probs": "prob"})
    df = df.drop(columns=["curr_token_prob"])
    rename_cols = [col for col in df.columns if col not in common_cols]
    if "base" not in col_prefix:
        drop_cols = [col for col in df.columns if col not in merge_cols + rename_cols]
        df = df.drop(columns=drop_cols)
    df = df.rename(columns={col: f"{col_prefix}{col}" for col in rename_cols})
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_results_dir", type=str, default="data/model_outputs/code/starcoder/apps/test/n_tkns_2e6/max_seq_len_64/seed_1")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="docs/data/code/starcoder/apps/test/n_tkns_2e6/max_seq_len_64/seed_1")
    args = parser.parse_args()
    model_results_dir = args.model_results_dir
    n_samples = args.n_samples
    output_dir = args.output_dir
    # Get all json files in the model_results_dir
    files = [f for f in os.listdir(model_results_dir) if f.endswith('.json') and "combined_results.json" not in f]

    
    common_cols = ["seq_id", "context_len", "prev_token", "curr_token", "in_tokens", "prev_token_id", "curr_token_id", "in_token_ids", "uniq_prev_tokens"]
    merge_cols = ["seq_id", "context_len"]
    
    merged_df = None

    # Read each json file and combine them into a single dataframe
    for file in files:
        col_prefix = file.split("model_outputs.json")[0]
        df = pd.read_json(os.path.join(model_results_dir, file), orient="records", lines=True)
        df = prep_df_for_merge(df, col_prefix, common_cols, merge_cols)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=merge_cols)

        print("Shape of df: ", df.shape)
        print("Shape of merged_df: ", merged_df.shape)
        
    # Save the merged dataframe
    merged_df.to_json(os.path.join(model_results_dir, "combined_results.json"), orient="records", lines=True)

    select_cols = ["seq_id", "context_len", "wrapped_context", "curr_token", "_prob", "_top_k_pred_tokens", "_top_k_pred_probs", "_entropy"]
    # Set seed for reproducibility
    np.random.seed(42)
    merged_df = merged_df.sample(n_samples)
    # Wrap input context 
    merged_df["wrapped_context"] = merged_df["in_tokens"].apply(
        lambda t: "<br>".join(textwrap.wrap(t))
    )
    select_cols = [col for col in merged_df.columns if any(c in col for c in select_cols)]
    merged_df = merged_df[select_cols]
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_json(os.path.join(output_dir, "combined_results_10000.json"), orient="records", lines=True)

if __name__ == "__main__":
    main()