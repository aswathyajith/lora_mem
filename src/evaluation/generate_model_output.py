from src.utils.generator import TokenGenerator
from src.utils.model import load_model
from src.utils.data import load_data
from datasets import Dataset
import argparse
import pandas as pd
import os
import shutil
from src.utils.model import compute_loss
import json

base_model_map = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b", 
    "pythia-160m": "EleutherAI/pythia-160m", 
}

def save_base_model_outputs(
        base_model_name: str,
        ds: Dataset,
        base_output_path: str
    ):
    """
    Compute and save base model outputs to a base_output_path json dataframe
    """
    os.makedirs(os.path.dirname(base_output_path), exist_ok=True)
    base_tkn_gen = TokenGenerator(base_model_name, lora_adapter_path=None)
    try:
        base_model_outputs = base_tkn_gen.iterate_over_ds(ds)
        base_model_outputs = pd.DataFrame(base_model_outputs)
        base_model_outputs.to_json(base_output_path, orient="records", lines=True)
    except Exception as e:
        raise(f"Error getting base model outputs for {base_model_name}: {e}")

    return base_model_outputs


def filter_configs(model_data_mapping: pd.DataFrame, domain: str):
    model_data_mapping = model_data_mapping[model_data_mapping["domain"] == domain]
    return model_data_mapping

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model_data_mapping", type=str, default="configs/model_data_train_config.csv")
    parser.add_argument("--output_dir", type=str, default="data/model_outputs/")
    parser.add_argument("--domain", type=str, default="code")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--get_ppl", action="store_true", default=False)
    args = parser.parse_args()

    output_dir_root = args.output_dir
    get_ppl = args.get_ppl
    model_data_mapping = pd.read_csv(args.path_to_model_data_mapping)
    model_data_mapping = filter_configs(model_data_mapping, args.domain)
    eval_split = "train" if "train" in args.path_to_model_data_mapping.split("/")[-1] else "test"

    for base_model in base_model_map:
        config_by_base_model = model_data_mapping[model_data_mapping["model_size"] == base_model]
        tokenizer = TokenGenerator(base_model_map[base_model], lora_adapter_path=None).tokenizer
        print(len(config_by_base_model))
        for _, row in config_by_base_model.iterrows():
            ft = row["ft"]
            model_path_root = row["model_path"]

            base_model = row["model_size"]
            base_model_name = base_model_map[base_model]
            domain = row["domain"]
            dataset = row["dataset_name"]
            max_seq_len = row["max_seq_len"]
            split = row["split"]

            # Load data
            ds = load_data(tokenizer, inference=True, **row)

            base_output_dir = os.path.join(output_dir_root, "base_model_outputs", base_model, domain, dataset, split, f"max_seq_len_{max_seq_len}")
            base_output_path = os.path.join(base_output_dir, "base_model_outputs.json")
            base_ppl_path = os.path.join(base_output_dir, "base_model_ppl.json")
            
            # Get base model outputs and save if it doesn't exist
            if not os.path.exists(base_output_path):
                save_base_model_outputs(base_model_name, ds, base_output_path)

            if get_ppl and not os.path.exists(base_ppl_path):
                ppl_args = {
                    "model_path_or_hf_name": base_model_name,
                    "lora_adapter_path": None
                }
                ppl = compute_loss(**ppl_args, dataset=ds, text_field=row["text_field"], max_seq_length=max_seq_len, packing=True, base_model=True)
                ppl_dict = {
                    'base_ppl': ppl
                }
                with open(base_ppl_path, 'w') as f:
                    json.dump(ppl_dict, f)
            
            for seed in args.seeds:
                
                model_path = os.path.join(model_path_root, f"seed_{seed}", "final_model")
                model_dataset = model_path.split("/" + domain + "/")[-1].split("/")[0]
    
                model_n_tkns = model_path.split("n_tkns_")[-1].split("/")[0]
                sub_dir_data = os.path.join(f"{domain}", model_dataset, f"{dataset}", eval_split, f"n_tkns_{model_n_tkns}", f"max_seq_len_{max_seq_len}", f"seed_{seed}")
                output_dir = os.path.join(output_dir_root, sub_dir_data)
                print("OUTPUT DIR", output_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                if ft == "full": 
                    filename = "full_model_outputs.json"
                    tkn_gen_args = {
                        "model_name": model_path,
                        "lora_adapter_path": None
                    }
                    if get_ppl:
                        ppl_args = {
                            "model_path_or_hf_name": model_path,
                            "lora_adapter_path": None
                        }
                        model_loss_name = "full_ppl"
                    
                else:                 
                    filename = f"lora_r{int(row['lora_rank'])}_model_outputs.json"
                            
                    tkn_gen_args = {
                        "model_name": base_model_name,
                        "lora_adapter_path": model_path
                    }
                    if get_ppl:
                        ppl_args = {
                            "model_path_or_hf_name": base_model_name,
                            "lora_adapter_path": model_path
                        }
                        model_loss_name = f"lora_r{int(row['lora_rank'])}_ppl"
                    
                if get_ppl:
                    path_to_ppls = os.path.join(output_dir, "losses.json")
                    loss_computed = False
                    ppl_dict = {}
                    if os.path.exists(path_to_ppls):
                        with open(path_to_ppls, 'r') as f:
                            ppl_dict = json.load(f)
                            if model_loss_name in ppl_dict:
                                print(f"Loss already exists for {model_loss_name}. Skipping...")
                                loss_computed = True
                    
                    if not loss_computed:
                        ppl = compute_loss(
                            **ppl_args, 
                            dataset=ds, 
                            text_field=row["text_field"], 
                            max_seq_length=max_seq_len, 
                            packing=True
                        )

                        ppl_dict[model_loss_name] = ppl
                        
                    if "base_ppl" not in ppl_dict:
                        # open base ppl
                        with open(base_ppl_path, 'r') as f:
                            base_ppl_dict = json.load(f)
                            base_ppl = base_ppl_dict["base_ppl"]
                            ppl_dict["base_ppl"] = base_ppl

                    with open(path_to_ppls, 'w') as f:
                        json.dump(ppl_dict, f)

                # Copy base model outputs to output dir
                base_path = os.path.join(output_dir, "base_model_outputs.json")
                if not os.path.exists(base_path):
                    shutil.copy(base_output_path, output_dir)

                model_outputs_path = os.path.join(output_dir, filename)
                if os.path.exists(model_outputs_path):
                    print(f"Model outputs already exist for {model_outputs_path}. Skipping...")
                    continue
                tkn_gen = TokenGenerator(**tkn_gen_args)
                
                # Generate model outputs
                model_outputs = tkn_gen.iterate_over_ds(ds)
                model_outputs = pd.DataFrame(model_outputs)

                # Save model outputs
                model_outputs.to_json(model_outputs_path, orient="records", lines=True)

                

if __name__ == "__main__": 
    main()
