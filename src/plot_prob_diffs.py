import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="docs/data/biomed/chemprot/seed_1/lora_r16_full_merged.json")
    parser.add_argument("--output_path", type=str, default="plots/biomed/chemprot/seed_1/lora_r16_full_merged.png")
    
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path
    data_path = "/net/projects/clab/aswathy/projects/lora_mem/docs/data"
    for root, dirs, files in os.walk(data_path):
        if not dirs:
            print('%s is a leaf' % root)
    exit()
    df = pd.read_json(data_path)
    output_dir = os.path.dirname(output_path)

    print(df.columns)
    df["full_base_diff"] = df["full_prob"] - df["base_prob"]
    df["lora_base_diff"] = df["lora_r16_prob"] - df["base_prob"]
    df["full_lora_diff"] = df["full_prob"] - df["lora_r16_prob"]

    # Plot probability differences
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="full_base_diff", y="lora_base_diff", data=df, hue="base_prob", alpha=0.5)

    plt.title("Divergence from base model probability")
    plt.xlabel(r"$p_{full} - p_{base}$")
    plt.ylabel(r"$p_{lora} - p_{base}$")
    plt.legend()
    plt.show()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    
    
    