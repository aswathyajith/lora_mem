import argparse 
import os
import json 
import pandas as pd
from src.utils.model import load_model, compute_loss
from src.utils.data import load_data

DOMAIN_DATASETS = {
    # "legal": ["us_bills"],
    # "biomed": ["chemprot"],
    "code": ["starcoder"],
    # "math": ["open_web_math"]
}

MODEL_DIR_TO_HF_NAME = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m"
}

def match_data_args(
        domain: str, 
        dataset_name: str, 
        model_dir: str, 
        preprocess_config_path: str, 
        n_tkns: int = 2e5,
    ) -> dict: 
    if not os.path.exists(preprocess_config_path): 
        raise FileNotFoundError(f"Preprocess config file {preprocess_config_path} does not exist")
    
    preprocess_config = pd.read_csv(preprocess_config_path)
        
    if "packing" in model_dir: 
        packing = True
    if "max_seq_len" in model_dir: 
        max_seq_len = int(model_dir.split("max_seq_len_")[-1].split("/")[0])
    if "/packing/" in model_dir: 
        packing = True 
        
    select_criteria = (
        (preprocess_config["domain"] == domain) & 
        (preprocess_config["dataset_name"] == dataset_name) & 
        (preprocess_config["max_seq_len"] == max_seq_len) & 
        (preprocess_config["n_tkns"] == n_tkns) & 
        (preprocess_config["packing"] == packing) & 
        (preprocess_config["split"] != "train")
    )
    print(f"Selection criteria: \n\t domain: {domain}\n\t dataset: {dataset_name}\n\t max_seq_len: {max_seq_len}\n\t n_tkns: {n_tkns}\n\t packing: {packing}\n\t split: (!= train)")

    selected_data_args = preprocess_config[select_criteria]
    # Drop duplicate rows
    selected_data_args = selected_data_args.drop_duplicates()
    print("# Matching selection criteria: ", len(selected_data_args))
    return selected_data_args.iloc[0]

def get_last_ckpt(path: str) -> str | None: 
    ckpts = [int(f.split("checkpoint-")[-1]) for f in os.listdir(path) if f.startswith("checkpoint")]
    if len(ckpts) == 0: 
        return None
    last_ckpt = max(ckpts)

    return os.path.join(path, f"checkpoint-{last_ckpt}")

def get_eval_loss(
        domain: str, 
        dataset_name: str, 
        model_path_or_hf_name : str, 
        lora_adapter_path : str | None = None,
        preprocess_config_path: str = "configs/preprocess_config.json",
        debug : bool = False
    ) -> float: 
    """
    Get eval loss from last checkpoint or final model
    """

    model_dir = lora_adapter_path if lora_adapter_path is not None else model_path_or_hf_name
    ft_complete = os.path.exists(model_dir)
    ckpt_dir = os.path.dirname(model_dir)
    if debug and not ft_complete: 
        return 0
    last_ckpt = get_last_ckpt(ckpt_dir)


    # Could not find any checkpoint, 
    # check if final_model dir contains results
    # else compute loss of final model
    if last_ckpt is None: 
        if os.path.exists(os.path.join(model_dir, "trainer_state.json")): 
            print(f"Found trainer_state.json in {model_dir}")
            last_ckpt = model_dir
        else:
            print(f"No trainer_state.json found in {model_dir}")
            # Load final model
            model, tokenizer = load_model(model_path_or_hf_name, lora_adapter_path)
            data_args = match_data_args(
                domain=domain, 
                dataset_name=dataset_name, 
                model_dir=model_dir, 
                preprocess_config_path=preprocess_config_path
            )
            print("Args to load_data: ", data_args)
            dataset = load_data(
                tokenizer,
                **data_args
            )

            sft_args = {
                "max_seq_length": data_args["max_seq_len"], 
                "packing": data_args["packing"], 
                "text_field": data_args["text_field"]
            }
            
            loss = compute_loss(
                model_path_or_hf_name=model_path_or_hf_name, lora_adapter_path=lora_adapter_path, 
                model=model, 
                dataset=dataset, 
                **sft_args)
            trainer_state = os.path.join(model_dir, "trainer_state.json")
            loss["best_metric"] = loss["eval_loss"]
            with open(trainer_state, "w") as f: 
                print(f"Writing evaluation results to {trainer_state}")
                json.dump(loss, f, indent=4)
            return loss["eval_loss"]
    
    trainer_state = os.path.join(last_ckpt, "trainer_state.json")
    
    with open(trainer_state, "r") as f: 
        eval_loss = json.load(f)
    return eval_loss["best_metric"]

def get_avg_loss(
        domain: str, 
        dataset_name: str, 
        model_name_or_path: str, 
        lora_adapter_path: str | None = None, 
        debug: bool = False, 
        seeds: list[int] = [1, 2, 3],
        preprocess_config_path: str = "config_dfs/configurations.csv"
    ) -> float:

    avg_loss = 0
    for i in seeds: 
        if lora_adapter_path is not None: 
            _model_name_or_path = model_name_or_path
            _lora_adapter_path = os.path.join(lora_adapter_path, f"seed_{i}", "final_model")
        else: 
            _model_name_or_path = os.path.join(model_name_or_path, f"seed_{i}", "final_model")
            _lora_adapter_path = lora_adapter_path
        
        ckpt_dir = os.path.exists(_model_name_or_path) if lora_adapter_path is None else os.path.exists(_lora_adapter_path)
        if not os.path.exists(ckpt_dir): 
            if debug: 
                continue
            else: 
                raise FileNotFoundError(f"Checkpoint dir {ckpt_dir} does not exist")
  
        loss = get_eval_loss(
            domain=domain, 
            dataset_name=dataset_name, 
            model_path_or_hf_name=_model_name_or_path, 
            lora_adapter_path=_lora_adapter_path, 
            preprocess_config_path=preprocess_config_path,
            debug=debug
        )
        avg_loss += loss
    avg_loss /= len(seeds)
    return avg_loss

def main(): 
    parser = argparse.ArgumentParser()
    # Example Usage: 
    # python src/find_opt_lr.py --all_models_path models/pythia-1.4b/packing_n_docs/perturbations/none --save_path configs/optimal_lr_n_docs.json --max_seq_lens 64 128 256
    parser.add_argument("--all_models_path", type=str, default="models/pythia-1.4b/packing/perturbations/none")
    parser.add_argument("--max_seq_lens", type=int, default=[64, 128, 256], nargs="+")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--preprocess_config_path", type=str, default="config_dfs/configurations.csv")
    parser.add_argument("--save_path", type=str, default="configs/optimal_lr.json")

    args = parser.parse_args()
    all_models_path = args.all_models_path
    debug = args.debug
    save_path = args.save_path
    preprocess_config_path = args.preprocess_config_path
    
    # List comprehension to get all seed model paths 
    model_paths = [root for (root, dirs, files) in os.walk(all_models_path) for dir in dirs if ("seed" in dir)]
    losses = []
    # Filter out model paths that are not in the predefined domain/datasets
    
    dd = [f"{k}/{v_i}" for k, v in DOMAIN_DATASETS.items() for v_i in v]

    model_paths = [path for path in model_paths if any(dd in path for dd in dd)]
    print("Found", len(model_paths), "model paths")

    for path in model_paths: 
        selected_domain = [domain for domain in DOMAIN_DATASETS.keys() if domain in path]
        domain = selected_domain[0]
        dataset_name = [ds for ds in DOMAIN_DATASETS[domain] if ds in path]
        dataset_name = dataset_name[0]
            
        if "full" in path: 
            model_name_or_path = path
            lora_adapter_path = None
        else: 
            model_name_or_path = MODEL_DIR_TO_HF_NAME[path.split("/")[1]]
            lora_adapter_path = path

        # Get loss for each model averaged across seeds
        losses.append(get_avg_loss(
            domain=domain, 
            dataset_name=dataset_name, 
            model_name_or_path=model_name_or_path, 
            lora_adapter_path=lora_adapter_path, 
            debug=debug, 
            preprocess_config_path=preprocess_config_path
        ))
    dict_losses = {
        model_path: loss for model_path, loss in zip(model_paths, losses)
    }
    path_hyperparams = {}
    
    for k, v in dict_losses.items(): 
        # Check if any of the predefined domains are in the model path
        selected_domain = [domain for domain in DOMAIN_DATASETS.keys() if domain in k]
        if len(selected_domain) == 0: 
            print(f"Skipping {k} because it is not in the predefined domains")
            continue
        else: 
            domain = selected_domain[0]
            selected_datasets = [ds for ds in DOMAIN_DATASETS[domain] if ds in k]
            if len(selected_datasets) == 0: 
                print(f"Skipping {k} because it is not in the predefined domains")
                continue
            
        dataset = selected_datasets[0]

        max_len = [l.replace("max_seq_len_", "") for l in k.split("/") if l.startswith("max_seq_len")][0]
        n_tkns = [n.replace("n_tkns_", "") for n in k.split("/") if n.startswith("n_tkns_")][0]
        model_size = k.split("/")[1]
        r = None
        if "full" in k: 
            ft = "full"
        else: 
            ft = "lora"
            r = [r.replace("r_", "") for r in k.split("lora/")[-1].split("/") if r.startswith("r_")][0]
        lr = [l.replace("lr_", "") for l in k.split("lora/")[-1].split("/") if l.startswith("lr_")][0]
        path_wo_lr = "/".join([k.split("/lr_")[0]] + k.split("/lr_")[-1].split("/")[1:])

        # print(domain_dataset, max_len, num_train, ft, r, lr, round(v, 2))
        if path_wo_lr not in path_hyperparams: 
            path_hyperparams[path_wo_lr] = {
                "model_size": model_size,
                "domain": domain,
                "dataset": dataset, 
                "max_seq_len": max_len, 
                "n_tkns": n_tkns, 
                "ft": ft, 
                "lora_rank": r, 
                "lr_loss_map": {
                    lr: (k, v)
                }
            }
        else: 
            path_hyperparams[path_wo_lr]["lr_loss_map"][lr] = (k, v)

    for model, hyperparams in path_hyperparams.items(): 
        # Select best lr for each model
        opt_lr, path_loss = min(hyperparams["lr_loss_map"].items(), key=lambda x: x[1][1])
        opt_model_path = path_loss[0]
        opt_loss = path_loss[1]
        path_hyperparams[model]["opt_lr"] = opt_lr
        path_hyperparams[model]["opt_loss"] = opt_loss
        path_hyperparams[model]["opt_model_path"] = opt_model_path
        del path_hyperparams[model]["lr_loss_map"]

    
    df = pd.DataFrame.from_dict(path_hyperparams, orient='index')
    df.reset_index(inplace=True)
    print(df.head())
    df = df[["model_size", "domain", "dataset", "max_seq_len", "n_tkns", "ft", "lora_rank", "opt_lr", "opt_loss", "opt_model_path"]]
    df.to_json(save_path, orient="records", indent=4)

if __name__ == "__main__": 
    main()