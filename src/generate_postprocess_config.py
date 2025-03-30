import argparse
import json
import os
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # python src/generate_postprocess_config.py --domain_datasets wiki/wikitext bible/bible_corpus_eng biomed/chemprot --max_seq_lens 64 128 256 --model pythia-1.4b --num_trains 4096 8192 16384 all --seeds 1 2 3 --config_path configs/postprocess_configs/pythia-1.4b.json --lora_ranks 16 --split validation
    parser.add_argument("--domain_datasets", type=str, required=True, nargs="+")
    parser.add_argument("--max_seq_lens", type=int, required=True, nargs="+")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_trains", type=str, required=True, nargs="+")
    parser.add_argument("--seeds", type=int, required=True, nargs="+")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lora_ranks", type=int, required=True, nargs="+")
    parser.add_argument("--data_dir", type=str, default="results/model_gens/pythia-1.4b/packing/perturbations/none/")
    parser.add_argument("--optimal_lr_path", type=str, default="configs/optimal_lr.json")
    args = parser.parse_args()
    domain_datasets = args.domain_datasets
    max_seq_lens = args.max_seq_lens
    num_trains = args.num_trains
    config_path = args.config_path
    seeds = args.seeds
    model = args.model
    lora_ranks = args.lora_ranks
    data_dir = args.data_dir
    optimal_lr_path = args.optimal_lr_path
    split = args.split
    optimal_lr_df = pd.read_json(optimal_lr_path, dtype={"opt_lr": object})

    sample_size = {
        "biomed/chemprot": "sample_all",
        "bible/bible_corpus_eng": "sample_2048",
        "wiki/wikitext": "sample_2048",
        "legal/cuad": "sample_2048",
    }

    processed_data_configs = {}
    for domain_dataset in domain_datasets: 
        for num_train in num_trains:
            # biomed/chemprot only has all num_train
            if domain_dataset == "biomed/chemprot" and num_train != "all":
                continue
            base_dir = f"{data_dir}/{domain_dataset}/{split}/base/num_train_{num_train}"
            full_dir = f"{data_dir}/{domain_dataset}/{split}/full-ft"
            lora_dir = f"{data_dir}/{domain_dataset}/{split}/lora"


            for max_seq_len in max_seq_lens:
                key = f"{domain_dataset}/{split}/num_train_{num_train}/max_seq_len_{max_seq_len}/{sample_size[domain_dataset]}"
                base_path = f"{base_dir}/max_seq_len_{max_seq_len}/{sample_size[domain_dataset]}/tkn_freq_probs.csv"
                opt_full_lr = optimal_lr_df[
                    (optimal_lr_df["model_size"] == model) &
                    (optimal_lr_df["domain_dataset"] == domain_dataset) &
                    (optimal_lr_df["num_train"] == num_train) &
                    (optimal_lr_df["ft"] == "full") & 
                    (optimal_lr_df["max_seq_len"] == max_seq_len)
                ]
                opt_full_lr = opt_full_lr["opt_lr"].values[0]
                full_path = f"{full_dir}/lr_{opt_full_lr}/num_train_{num_train}/max_seq_len_{max_seq_len}/{sample_size[domain_dataset]}"
                for lora_rank in lora_ranks:
                    opt_lora_lr = optimal_lr_df[
                        (optimal_lr_df["model_size"] == model) &
                        (optimal_lr_df["domain_dataset"] == domain_dataset) &
                        (optimal_lr_df["num_train"] == num_train) &
                        (optimal_lr_df["ft"] == "lora") & 
                        (optimal_lr_df["lora_rank"] == lora_rank) &
                        (optimal_lr_df["max_seq_len"] == max_seq_len)
                    ]
                    opt_lora_lr = opt_lora_lr["opt_lr"].values[0]
                    lora_path = os.path.join(lora_dir, f"r_{lora_rank}", f"lr_{opt_lora_lr}", f"num_train_{num_train}", f"max_seq_len_{max_seq_len}",sample_size[domain_dataset])

                    processed_data_configs[key] = {
                        "base": {
                            "path": base_path
                        },

                        f"lora_r{lora_rank}": {
                            "path": lora_path
                        },

                        "full": {
                            "path": full_path
                        }
                    }
                    # debug
                    if not os.path.exists(base_path):
                        print(f"base_path: {base_path} DOES NOT EXIST. EXITING!")
                        exit()
                    if not os.path.exists(full_path):
                        print(f"full_path: {full_path} DOES NOT EXIST. EXITING!")
                        exit()  
                    if not os.path.exists(lora_path):
                        print(f"lora_path: {lora_path} DOES NOT EXIST. EXITING!")
                        exit()


    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(processed_data_configs, f, indent=4)
