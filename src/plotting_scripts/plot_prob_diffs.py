import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
import re
from matplotlib.colors import LogNorm

# domain_dataset="biomed/chemprot"
# domain_dataset="bible/bible_corpus_eng"
# train_size="num_train_4096"
# python src/plotting_scripts/plot_prob_diffs.py --data_path data/plotting_data/pythia-1.4b/packing/perturbations/none --output_dir plots/pythia-1.4b/perturbations/none/all_datasets/validation/r16 --lora_ranks 16 --split validation
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="docs/data")#/biomed/chemprot/seed_1/lora_r16_full_merged.json")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--single_dataset", default=False, action="store_true")
    parser.add_argument("--kde", default=False, action="store_true")
    parser.add_argument("--seed", type=str, default="1")
    parser.add_argument("--domain_dataset", type=str, default="all")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lora_ranks", type=str, default=["16", "256"], nargs="+")
    parser.add_argument("--val_loss_path", type=str, default="configs/optimal_lr.json")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    out_file = args.out_file
    ppl_path = "results/model_ppls.csv"
    kde = args.kde
    single_dataset = args.single_dataset
    seed = args.seed
    domain_dataset = args.domain_dataset
    split = args.split
    lora_ranks = args.lora_ranks
    val_loss_path = args.val_loss_path

    if not out_file:
        out_filename = "prob_diffs"
    else:
        out_filename = out_file.replace(".pdf", "").replace(".png", "")

    n = 0
    in_files = []
    for root, dirs, files in os.walk(data_path):
        if not dirs:
            dir_files = [os.path.join(os.path.relpath(root), f) for f in files if ((f.endswith(".json") and (f"seed_{seed}" in root)))]
            n += len(dir_files)
            in_files.extend(dir_files)
    
    # if single_dataset:
    #     # Filter in_files to only include files with num_train_all
    #     in_files = [f for f in in_files if "num_train_4096" in f]

    #     # Sort in_files by max_seq_len and num_train
    #     in_files = sorted(in_files, key=lambda x: (int(x.split("/")[5].split("num_train_")[1]), int(x.split("/")[6].split("max_seq_len_")[1])))

        
    
    import re

    def natural_sort_key(text):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', text)]

    in_files = sorted(in_files, key=natural_sort_key)
    
    in_files = [f for f in in_files if any(f"lora_r{r}" in f for r in lora_ranks)]
    n = len(in_files)
    print(f"Found {n} files")

    ppl_map = pd.read_json(val_loss_path)

    # ppl_df = pd.read_csv(ppl_path)

    # ppl_map = {
    #     "wiki/wikitext" : "wikitext:wikitext-2-raw-v1", 
    #     "bible/bible_corpus_eng" : "davidstap/biblenlp-corpus-mmteb:eng-arb", 
    #     "biomed/chemprot" : "bigbio/chemprot:chemprot_full_source", 
    #     "legal/us_bills" : "pile-of-law/pile-of-law:us_bills", 
    #     "math/open_web_math" : "open-web-math/open-web-math", 
    #     "code/starcoder" : "lparkourer10/starcoder-python5b"
    # }
    # Define subplot size
    ncols = 3
    if lora_ranks == ["256"]:
        ncols = 2
    n_subplots = ncols * (math.ceil(n / ncols))
    nrows = int(n_subplots / ncols)
    f, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3)) 

    # First pass: calculate all densities and find global max
    all_counts = []
    for file in in_files:
        df = pd.read_json(file)
        lora_rank = file.split('lora_r')[-1].split('_')[0]
        print(file, lora_rank)
        df["full_base_diff"] = df["full_prob"] - df["base_prob"]
        df["lora_base_diff"] = df[f"lora_r{lora_rank}_prob"] - df["base_prob"]
        df["full_lora_diff"] = df["full_prob"] - df[f"lora_r{lora_rank}_prob"]
        nan = df[df["full_base_diff"].isna() | df["lora_base_diff"].isna()][["curr_token", "full_prob", f"lora_r{lora_rank}_prob"]]
        if len(nan) > 0:
            print(nan.head())
            exit()
        if single_dataset:
            L = int(file.split("/")[6].split("max_seq_len_")[1])
        
        x = df["full_base_diff"]
        y = df["lora_base_diff"]
        
        # Calculate 2D histogram
        hist, _, _ = np.histogram2d(x, y, bins=50, range=[[-1, 1], [-1, 1]])
        all_counts.append(hist)
    
    # Find global max for normalization
    vmax = max(hist.max() for hist in all_counts)
    norm = LogNorm(vmin=1, vmax=vmax)
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, as_cmap=True)
    print(vmax)

    axs = axs.flatten()
    
    for file, ax in zip(in_files, axs):
        df = pd.read_json(file)

        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgrey')

        # if single_dataset:
        #     N = int(file.split("/")[5].split("num_train_")[1])
        #     L = int(file.split("/")[6].split("max_seq_len_")[1])
        #     title = f"N={N}, L={L}"
            # if L==1024:
            #    df =  df.sample(1000)
        

        print(len(df))

        output_path = file.replace(data_path, output_dir).replace(".json", ".png")
        
        print(f"Plotting {file} to {output_path}")
        lora_rank = file.split('lora_r')[-1].split('_')[0]

        df["full_base_diff"] = df["full_prob"] - df["base_prob"]
        df["lora_base_diff"] = df[f"lora_r{lora_rank}_prob"] - df["base_prob"]
        df["full_lora_diff"] = df["full_prob"] - df[f"lora_r{lora_rank}_prob"]

        df_blw_thresh = df[abs(df["full_lora_diff"]) < 0.1]

        if kde:
            # sns.kdeplot(x="full_base_diff", y="lora_base_diff", data=df, fill=True, ax=ax)

            # Calculate 2D histogram
            x = df["full_base_diff"]
            y = df["lora_base_diff"]
            # hist, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[-1, 1], [-1, 1]])
            # First plot KDE
            # sns.kdeplot(x=x, y=y, levels=5, color="blue", 
            #             fill=True, alpha=0.3, ax=ax)
            # # Then overlay scatter
            # sns.scatterplot(x=x, y=y, data=df, alpha=0.2, 
            #                 s=0.7, color="black", ax=ax)
            
            hist = sns.histplot(x=x, y=y, bins=10, cmap=cmap, ax=ax, cbar=False, stat='count')
            # Get the histogram data from the plot

            # # Update the normalization
            # hist.set_norm(norm)

             # Get the count values
            counts = hist.collections[0].get_array()
            print(counts)
            print(f"Number of NaN values in x: {x.isna().sum()}")
            print(f"Number of NaN values in y: {y.isna().sum()}")
            print(f"For subplot {ax.get_title()}:")
            print(f"Max count: {counts.max()}")
            print(f"Min count: {counts.min()}")
            print(f"Mean count: {counts.mean():.2f}")
            print(f"Median count: {np.median(counts):.2f}")
            print(f"Total count: {counts.sum()}")
            print("---")
        else:
            sns.scatterplot(x="full_base_diff", y="lora_base_diff", data=df, hue="base_prob", ax=ax, s=0.7, edgecolor=None)
        
            # color points below threshold in red
            if df_blw_thresh.shape[0] > 0:
                sns.scatterplot(x="full_base_diff", y="lora_base_diff", data=df_blw_thresh, ax=ax, s=0.7, edgecolor=None)

        # print PPL if computed
        # if not single_dataset:
        #     dataset = ppl_map["/".join(file.split("/")[3:5])]
        #     if dataset in ppl_df["dataset"].values:
        #         ppl = ppl_df[ppl_df["dataset"] == dataset]["perplexity"].values[0]
        #         ax.text(-0.9, 0.8,f"PPL: {ppl:.2f}", fontsize=7.5, color="red")
        #         title = f"{'/'.join(file.split('/')[3:5])}"

        
        domain_dataset_name = "/".join(file.split("perturbations")[-1].split(split)[0].strip("/").split("/")[1:])
        
        title = f"Dataset: {domain_dataset_name}"
        max_seq_len = [i for i in file.split('/') if "max_seq_len" in i][0]
        max_seq_len_str = max_seq_len.replace("max_seq_len_", "Context Length: ")
        num_train = [i for i in file.split('/') if "num_train_" in i][0]
        num_train_str = num_train.replace("num_train_", "Train Size: ")
        lora_rank = file.split('lora_r')[-1].split('_')[0]
        lora_rank_str = f"LoRA Rank: {lora_rank}" 
        ppl_key_prefix = file.split(domain_dataset_name)[0]
        # print(file, domain_dataset_name)
        ppl_key_prefix = ppl_key_prefix.replace("data/plotting_data", "models") 
        full_ppl_key = f"{ppl_key_prefix + domain_dataset_name}/full-ft/{num_train}/{max_seq_len}"
        lora_ppl_key = f"{ppl_key_prefix + domain_dataset_name}/lora/r_{lora_rank}/{num_train}/{max_seq_len}"
        full_ppl = ppl_map[ppl_map["index"] == full_ppl_key]["opt_loss"].iloc[0]
        lora_ppl = ppl_map[ppl_map["index"] == lora_ppl_key]["opt_loss"].iloc[0]
        full_ppl_str = f"Full PPL: {full_ppl:.2f}"
        lora_ppl_str = f"LoRA PPL: {lora_ppl:.2f}"

        title = "\n".join([title, num_train_str, max_seq_len_str, lora_rank_str, full_ppl_str, lora_ppl_str])
        ax.set_title(title)
        ax.set_xlabel(r"$p_{full} - p_{base}$")
        ax.set_ylabel(r"$p_{lora} - p_{base}$")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        
        # move legend outside of plot
        if not kde:
            ax.legend(title=r"$p_{base}$", loc="upper left", bbox_to_anchor=(1, 1))
            # make legend smaller
            legend = ax.get_legend()
            # reduce legend size
            plt.setp(legend.get_frame(), width=0.5, height=0.5)
            
            # change legend marker size
            for handle in legend.legend_handles:
                handle.set_markersize(5)
        
    # axes to be removed
    if n % n_subplots != 0:
        for ax in range(n_subplots-(n % n_subplots)):
            f.delaxes(axs[-(ax+1)])
    
    if single_dataset:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=axs.ravel().tolist(), label='Count')

    f.suptitle(r"Divergence of $p_{lora}$ and $p_{full}$ from $p_{base}$", x=0.5, y=0.9975)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)

    # plt.savefig(os.path.join(output_dir, f"{out_file}.pdf"))
    out_file = "prob_diffs"
    plt.savefig(os.path.join(output_dir, f"{out_file}.png"), bbox_inches="tight")
    print("Saving plot as pdf")
    plt.savefig(os.path.join(output_dir, f"{out_file}.pdf"))
    
    
    
    