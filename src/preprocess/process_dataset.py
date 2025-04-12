import os
from src.utils.data import get_tkn_counts, get_pair_counts, load_data
from src.utils.model import load_model
import argparse
import json
import pandas as pd

def filter_config_df(df, indices_list, domains, datasets, split_name, **kwargs):
    """
    Filter the config_df based on the indices, domains, datasets, and split_name.
    """
    if len(indices_list) > 0:
        print("Filtering config_df based on indices")
        df = df.iloc[indices_list]

    if domains is not None:
        print("Filtering config_df for domains in", domains)
        df = df[df["domain"].isin(domains)]
        if datasets is not None:
            print("Filtering config_df for datasets in", datasets)
            df = df[df["dataset_name"].isin(datasets)]

    if split_name is not None:
        print("Filtering config_df for split_name in", split_name)
        df = df[df["eval_on_split"] == split_name]

    return df

def construct_save_dir(path_to_save_freqs, kwargs):
    packing = "packing" if kwargs["packing"] else "no_packing"
    save_dir = os.path.join(path_to_save_freqs, packing, "perturbations", kwargs["perturbation"], kwargs["split"], kwargs["domain"], kwargs["dataset_name"], f"max_seq_len_{kwargs['max_seq_len']}", f"n_tkns_{kwargs['n_tkns']}")
    return save_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="EleutherAI/pythia-1.4b", help="Base model to use for tokenization")
    parser.add_argument("--config_df_path", type=str, help="Path to the config_df csv file", default="config_dfs/configurations.csv")
    parser.add_argument("--path_to_save_freqs", type=str, help="Path to save the token and pair counts", default="data/processed/tkn_freqs")
    parser.add_argument("--domains", type=str, default=None, help="Domains to process", nargs="+")
    parser.add_argument("--datasets", type=str, default=None, help="Datasets to process (for this to work, domain must be specified)", nargs="+")
    parser.add_argument("--indices_list", type=int, help="List of indices to process in the config_df csv (default: all)", nargs="+", default=[])
    parser.add_argument("--split_name", type=str, default=None)

    args = parser.parse_args()
    
    config_df_path = args.config_df_path
    path_to_save_freqs = args.path_to_save_freqs
    filter_args = vars(args)

    # Load config 
    print("Loading config_df")
    config_df = pd.read_csv(
        config_df_path, dtype={
            "streaming": bool, 
            "skip": bool, 
            "n_tkns": str, 
        }
    )
    print("Example n_tkns: ", config_df.n_tkns.values[0])
    print(f"Total number of configurations: {len(config_df)}")
    # Filter config_df for args passed
    config_df = filter_config_df(config_df, **filter_args)
    print(f"Number of configurations after filtering: {len(config_df)}")

    for _, row in config_df.iterrows():
        kwargs = row.to_dict()
        kwargs["perturbation"] = "none" if kwargs["perturbation"] is None else kwargs["perturbation"]
        kwargs["packing"] = "packing" if kwargs["packing"] else "no_packing"
        save_dir = construct_save_dir(path_to_save_freqs, kwargs)
        tkn_counts_save_path = os.path.join(save_dir, "tkn_counts.csv")
        pair_counts_save_path = os.path.join(save_dir, "pair_counts.csv")

        if os.path.exists(tkn_counts_save_path) and os.path.exists(pair_counts_save_path):
            print(f"Skipping {kwargs['domain']}/{kwargs['dataset_name']} because it already exists")
            continue

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Load tokenizer and dataset 
        _, tokenizer = load_model(args.base_model, lora_adapter_path = None)
        dataset = load_data(tokenizer, **kwargs)

        # Get token counts 
        if not os.path.exists(tkn_counts_save_path):
            print(f"Getting token counts for {kwargs['domain']}/{kwargs['dataset_name']}...")
            tkn_counts = get_tkn_counts(dataset)
            tkn_counts.to_csv(tkn_counts_save_path, index=False)
            print(f"Saved token counts to {tkn_counts_save_path}")

        # Get pair counts 
        if not os.path.exists(pair_counts_save_path):
            print(f"Getting pair counts for {kwargs['domain']}/{kwargs['dataset_name']}...")
            pair_counts = get_pair_counts(dataset)
            pair_counts.to_csv(pair_counts_save_path, index=False)
            print(f"Saved pair counts to {pair_counts_save_path}")

if __name__ == "__main__":
    main()