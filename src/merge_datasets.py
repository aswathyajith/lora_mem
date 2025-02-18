import pandas as pd
import argparse
import random
import textwrap
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/output_token_info/pythia-1.4b/legal")
    parser.add_argument("--output_dir", type=str, default="docs/data")
    parser.add_argument("--model1", type=str, default="lora")
    parser.add_argument("--model2", type=str, default="full")
    parser.add_argument("--model1_seed", type=int, default=1)
    parser.add_argument("--model2_seed", type=int, default=1)
    parser.add_argument("--model1_r", type=int, default=16)
    parser.add_argument("--model2_r", type=int, default=16)
    args = parser.parse_args()
    data_path = args.data_path
    model1 = args.model1
    model2 = args.model2
    model1_seed = args.model1_seed
    model2_seed = args.model2_seed
    output_dir = args.output_dir
    model1_prefix = model1 + (f"_r{args.model1_r}" if model1 == "lora" else "")
    model2_prefix = model2 + (f"_r{args.model2_r}" if model2 == "lora" else "")
    
    # Check if merged outputs already exist
    output_path = os.path.join(args.output_dir, f"seed_{model1_seed}", f"{model2_prefix}_{model1_prefix}_merged.csv")
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Exiting.")
        
        exit()

    output_path = os.path.join(args.output_dir, f"seed_{model1_seed}", f"{model1_prefix}_{model2_prefix}_merged.json")
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Exiting")
        exit()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_model_path = os.path.join(data_path, f"seed_{model1_seed}/base_processed.csv")
    model1_path = os.path.join(data_path, f"seed_{model1_seed}/{model1_prefix}_processed.csv")
    model2_path = os.path.join(data_path, f"seed_{model2_seed}/{model2_prefix}_processed.csv")

    print("Base model path: ", base_model_path)
    print("Model 1 path: ", model1_path)
    print("Model 2 path: ", model2_path)

    base_df = pd.read_csv(base_model_path)
    model1_df = pd.read_csv(model1_path)
    model2_df = pd.read_csv(model2_path)

    if model1_prefix == model2_prefix: 
        model1_prefix = model1_prefix + f"_{model1_seed}"
        model2_prefix = model2_prefix + f"_{model2_seed}"

    # Subset to relevant cols
    sel_cols = ["seq_id", "in_tokens", "curr_token_prob", "curr_token_rank", "top_k_pred_tokens", "top_k_pred_probs"]
    model1_df = model1_df[sel_cols]
    model2_df = model2_df[sel_cols]

    # Rename cols to include prefix
    def rename_cols(df, prefix):
        df = df.rename(columns={
            "curr_token_prob": f"{prefix}_prob", 
            "curr_token_rank": f"{prefix}_rank", 
            "top_k_pred_tokens": f"{prefix}_top_k_pred_tokens", 
            "top_k_pred_probs": f"{prefix}_top_k_pred_probs"
        })
        return df
    
    base_df = rename_cols(base_df, "base")
    model1_df = rename_cols(model1_df, model1_prefix)
    model2_df = rename_cols(model2_df, model2_prefix)

    # Merge dfs
    print(f"Merging base with {model1}")

    merged_df = base_df.merge(model1_df, on=["seq_id", "in_tokens"])
    print(f"Merging with {model2}")
    merged_df = merged_df.merge(model2_df, on=["seq_id", "in_tokens"])
    merged_df["ft_prob_diff"] = merged_df[f"{model1_prefix}_prob"] - merged_df[f"{model2_prefix}_prob"]
    merged_df = merged_df[abs(merged_df["ft_prob_diff"]) > 0.1]
    seq_ids = list(set(merged_df.seq_id))
    random.seed(1)
    sample_frac = 0.1
    sample_size = int(len(seq_ids) * sample_frac)
    seq_ids = random.sample(seq_ids, sample_size)
    merged_df = merged_df[merged_df.seq_id.isin(seq_ids)]
    # print(len(merged_df))
    merged_df["wrapped_context"] = merged_df["in_tokens"].apply(
        lambda t: "<br>".join(textwrap.wrap(t))
    )
    merged_df = merged_df[["wrapped_context", "curr_token", "base_prob", "lora_r16_prob", "full_prob", "ft_prob_diff", "rel_prev"]]
    merged_df.to_json(output_path, orient="records")