import json 
import pandas as pd
import itertools
import os
path_to_preprocess_config = "configs/preprocess_config.json"
path_to_save_configs = "config_dfs/configurations.csv"

with open(path_to_preprocess_config, "r") as f:
    preprocess_config = json.load(f)

max_seq_len = preprocess_config["max_seq_len"]
n_tkns = preprocess_config["n_tkns"]
n_test_tkns = preprocess_config["n_test_tkns"]
packing = preprocess_config["packing"]
perturbations = preprocess_config["perturbations"]

configurations_list = []

for domain, datasets in preprocess_config["domains"].items():
    print(f"Generating configs for {domain}")
    for dataset_config in datasets:
        dataset_name = dataset_config["dataset_name"]
        if dataset_config.get("skip", False):
            continue

        splits = dataset_config.get("splits", {})
        

        eval_on_splits = dataset_config.get("eval_on_splits", [])
        if len(eval_on_splits) == 0:
            print(f"No splits to evaluate on for {dataset_name}")
            continue
            
        eval_split_names = []
        for split in eval_on_splits:
            if split not in splits:
                print(f"Split {split} not in {dataset_name}")
                continue

            else: 
                split_name = splits[split]
                eval_split_names.append(split_name)

        streaming = dataset_config.get("streaming", False)
        downloaded = dataset_config.get("downloaded", False)
        text_field = dataset_config["text_field"]
        dataset = dataset_config["dataset_name"] if downloaded else dataset_config["dataset"]

        curr_dataset = {
            "domain": domain,
            "dataset": dataset,
            "dataset_name": dataset_name,
            "text_field": text_field,
            "split": eval_split_names,
            "streaming": streaming, 
            "downloaded": downloaded
        }

        # Convert all values to lists to apply itertools.product
        for k, v in curr_dataset.items():
            if not isinstance(v, list):
                curr_dataset[k] = [v]

        product = list(itertools.product(*curr_dataset.values()))
        for prod_i in product:
            dict_entry = dict(zip(curr_dataset.keys(), prod_i))
            print(dict_entry)
            configurations_list.append(dict_entry)
    
# Cross with max_seq_len
settings = {
    "max_seq_len": max_seq_len,
    "n_tkns": n_tkns,
    "packing": packing,
    "perturbation": perturbations
}

for i, setting in settings.items():
    if not isinstance(setting, list):
        settings[i] = [setting]

data_settings = list(itertools.product(*settings.values()))
data_settings = [dict(zip(settings.keys(), data_setting)) for data_setting in data_settings]

configurations_list = [
    {**config, **data_setting}
    for config in configurations_list
    for data_setting in data_settings
]

configurations_df = pd.DataFrame(configurations_list)
configurations_df.loc[configurations_df["split"] != "train", "n_tkns"] = n_test_tkns
os.makedirs(os.path.dirname(path_to_save_configs), exist_ok=True)
configurations_df.to_csv(path_to_save_configs, index=False)