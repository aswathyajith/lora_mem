import argparse
import os 
import pandas as pd 
import torch

SPLIT_MERGE_COLS = {
    "train": ["domain", "dataset_name", "max_seq_len"],
    "test": ["domain", "max_seq_len"]
}

def split_train_test(dataset_configs):
    dataset_configs["n_tkns"] = 2e5 # Hardcoded number of tokens to use for inference/eval
    dataset_configs = dataset_configs.drop_duplicates()
    train_dataset_configs = dataset_configs[dataset_configs["split"] == "train"]
    test_dataset_configs = dataset_configs[dataset_configs["split"] != "train"]
    print(f"Split train dataset configs shape: {train_dataset_configs.shape}")
    print(f"Split test dataset configs shape: {test_dataset_configs.shape}")
    return {"train": train_dataset_configs, "test": test_dataset_configs}

def merge_configs(dataset_configs, model_configs):
    """
    Merge dataset and model configs on the split columns
    """
    data_config_splits = split_train_test(dataset_configs)
    for split in ["train", "test"]:
        data_configs = data_config_splits[split]
        model_configs = model_configs[SPLIT_MERGE_COLS[split] + ["model_size", "ft", "lora_rank", "model_path"]]
        merged_configs = pd.merge(data_configs, model_configs, on=SPLIT_MERGE_COLS[split])
        merged_configs.to_csv(f"configs/model_data_{split}_config.csv", index=False)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_configs_csv", type=str, default="config_dfs/configurations.csv")
    parser.add_argument("--model_selection_path", type=str, default="configs/optimal_lr.json")
    # TODO: Add subparsers for filtering

    args = parser.parse_args()
    dataset_config_path = args.dataset_configs_csv
    model_config_path = args.model_selection_path
    
    # Load configurations
    dataset_configs = pd.read_csv(dataset_config_path)
    model_configs = pd.read_json(model_config_path)

    # Filter configurations
    # TODO: Filter data and model dfs


    # Rename "dataset" to "dataset_name" in model_configs
    model_configs = model_configs.rename(columns={"dataset": "dataset_name", "opt_model_path": "model_path"})
    merge_configs(dataset_configs, model_configs)

    # TODO: Filter merged df
    

if __name__ == "__main__": 
    main()