import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
from scipy import integrate

# def nat_sort(s):
#     s = [x for x in s.split("/") if "num_train" in x][0].split("_")[-1]
#     # s = s.replace("all", "-1")
#     if s == "all":
#         return np.inf
#     return int(s)

def nat_sort(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', text)]

def prep_plotting_data(file_path):
    df = pd.read_json(file_path)
    df["full_minus_base"] = df.full_prob - df.base_prob
    df["lora_minus_base"] = df.lora_r16_prob - df.base_prob
    
    df = pd.read_json(file_path)
    df["full_minus_lora"] = df.full_prob - df.lora_r16_prob
    df["full_minus_base"] = df.full_prob - df.base_prob
    df["lora_minus_base"] = df.lora_r16_prob - df.base_prob

    y_values = []
    x_values = []
    diff_thresholds = np.linspace(0, 1, 100)

    for x in diff_thresholds:
        y = df[(df.full_minus_lora >= x) & (df.full_minus_base > 0.0) & (df.lora_minus_base > 0.0)]
        y_values.append(len(y))
        x_values.append(x)
    return [x_values, y_values]

# plot change in probs for each quantile
def plot_quantile_change(file_path, label, ax=None):
    df = pd.read_json(file_path)
    df["full_minus_base"] = df.full_prob - df.base_prob
    df["lora_minus_base"] = df.lora_r16_prob - df.base_prob
    df["full_minus_lora"] = df.full_prob - df.lora_r16_prob
    df = df[(df.full_minus_lora >= 0) & (df.full_minus_base > 0.0) & (df.lora_minus_base > 0.0)]
    
    # Use fixed bins from 0 to 1
    bins = np.linspace(0, 1, 100)
    hist, bin_edges = np.histogram(df.full_minus_lora.values, bins=bins, density=True)
    dx = bin_edges[1] - bin_edges[0]
    cdf = np.cumsum(hist * dx)
    auc = np.trapz(y=cdf, x=bin_edges[1:])
    
    # Get the plotting function based on whether ax is provided
    plot_fn = ax.ecdf if ax is not None else plt.ecdf
    
    # Plot using the appropriate function
    plot_fn(df.full_minus_lora, label=f"{label} (AUC: {auc:.2f})")
    if ax is not None:
        ax.set_ylabel(r"Proportion of tokens")
        ax.set_xlabel(r"$p_{full} - p_{lora}$")
        # ax.legend()
        # make legend smaller (not font size)
        ax.legend(prop={'size': 7}, loc='lower right')
    else:
        plt.ylabel(r"Proportion of tokens")
        plt.xlabel(r"$p_{full} - p_{lora}$")
        plt.legend()

def plot_change(x_values, y_values, label):
    plt.plot(x_values, y_values, label=label)
    plt.xlabel(r"$p_{full} - p_{lora}$")
    plt.ylabel(r"Number of tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/plotting_configs/val_cdf.json")
    parser.add_argument("--combined_cdfs_save_dir", type=str, default="plots/combined/validation")

    # parser.add_argument("--file_paths", type=str, default=["data/plotting_data/pythia-1.4b/packing/wiki/wikitext/train/num_train_all/max_seq_len_128/sample_2048/seed_1/lora_r16_full_merged.json", "data/plotting_data/pythia-1.4b/packing/wiki/wikitext/train/num_train_all/max_seq_len_256/sample_2048/seed_1/lora_r16_full_merged.json"], nargs="+")
    # parser.add_argument("--save_path", type=str, default="plots/pythia-1.4b/wiki/wikitext/train/num_train_all/seed_1/divergence_change.png")

    # parser.add_argument("--file_paths", type=str, default=["data/plotting_data/pythia-1.4b/packing/biomed/chemprot/train/num_train_all/max_seq_len_128/sample_all/seed_1/lora_r16_full_merged.json", "data/plotting_data/pythia-1.4b/packing/biomed/chemprot/train/num_train_all/max_seq_len_256/sample_all/seed_1/lora_r16_full_merged.json"], nargs="+")
    # parser.add_argument("--save_path", type=str, default="plots/pythia-1.4b/biomed/chemprot/train/num_train_all/seed_1/divergence_change.png")
    args = parser.parse_args()
    combined_cdfs_save_dir = args.combined_cdfs_save_dir
    os.makedirs(combined_cdfs_save_dir, exist_ok=True)
    # os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Group file paths by seed
    seed_files = {}
    for save_path, file_paths in config.items():
        seed = int([x for x in save_path.split("/") if "seed" in x][0].split("_")[-1])
        if seed not in seed_files:
            seed_files[seed] = {save_path: []}
        if save_path not in seed_files[seed]:
            seed_files[seed][save_path] = []
        for file_path in file_paths:
            seed_files[seed][save_path].append(file_path)

    # Create individual plots first
    # for save_path, file_paths in config.items():
    #     plt.figure(figsize=(10, 6))
    #     for file_path in file_paths:
    #         label = [x for x in file_path.split("/") if "max_seq_len" in x][0].split("_")[-1]
    #         plot_quantile_change(file_path, label)
    #     plt.savefig(save_path)
    #     plt.close()
    #     print(f"Saved individual plot to {save_path}")

    # Create subplot figure for all seeds
    n_seeds = len(seed_files)
    print(seed_files.keys())
    
    n_datasets = 3 # num rows
    n_total_trains = 4 # num cols
    row_ids = {
        "bible/bible_corpus_eng": 0,
        "wiki/wikitext": 1,
        "biomed/chemprot": 2,
    }
    col_ids = {
        "num_train_4096": 0,
        "num_train_8192": 1,
        "num_train_16384": 2,
        "num_train_all": 3,
    }
    for seed, files in seed_files.items():
        n_files = len(files)
        # fig, axs = plt.subplots(n_files, 1, figsize=(6, 6*n_files))
        # fig, axs = plt.subplots(n_datasets, n_total_trains, figsize=(6*n_datasets, 6*n_total_trains))
        fig, axs = plt.subplots(n_datasets, n_total_trains, figsize=(3*len(col_ids), 3*len(row_ids)))
        # axs = axs.flatten()
        sorted_files = sorted(files.items(), key=lambda x: nat_sort(x[0]))
        all_axs_idx = [
            (i, j) for i in range(n_datasets) for j in range(n_total_trains)
        ]
        print(all_axs_idx)
        
        for i, (save_path, file_paths) in enumerate(sorted_files):
            dd = "/".join(save_path.split("/")[2:4])
            domain_dataset = f'Dataset: {dd}'
            num_train = f'# train examples: {[x for x in save_path.split("/") if "num_train" in x][0].split("_")[-1]}'
            
            title = "\n".join([domain_dataset, num_train])
            for file_path in file_paths:
                label = [x for x in file_path.split("/") if "max_seq_len" in x][0].split("_")[-1]
                num_train = [x for x in save_path.split("/") if "num_train" in x][0].split("_")[-1]
                
                row_id = row_ids[dd]
                col_id = col_ids[f"num_train_{num_train}"]
                
                print(row_id, col_id)
                if (row_id, col_id) in all_axs_idx:
                    # Remove ax_idx from all_axs_idx and set title if data is present
                    all_axs_idx.remove((row_id, col_id)) 
                    axs[row_id, col_id].set_title(title)
                plot_quantile_change(file_path, label, ax=axs[row_id, col_id])
        
        for (row_id, col_id) in all_axs_idx:
            # Remove the axes 
            axs[row_id, col_id].set_axis_off()
            
        save_path = os.path.join(combined_cdfs_save_dir, f"seed_{seed}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved seed {seed} plot")
        exit()

        