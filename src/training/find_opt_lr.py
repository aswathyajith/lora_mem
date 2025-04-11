import argparse 
import os
import json 
import pandas as pd
from utils.model import load_model, compute_loss
from utils.data import load_data

DOMAIN_DATASETS = [
    "wiki/wikitext",
    "biomed/chemprot",
    "bible/bible_corpus_eng", 
    "legal/pile-of-law:us_bills",
    "code/starcoder",
    "math/open_web_math",
]

MODEL_DIR_TO_HF_NAME = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m"
}

def get_last_ckpt(path): 
    ckpts = [int(f.split("checkpoint-")[-1]) for f in os.listdir(path) if f.startswith("checkpoint")]
    if len(ckpts) == 0: 
        return None
    last_ckpt = max(ckpts)

    return os.path.join(path, f"checkpoint-{last_ckpt}")

def get_eval_loss(
        model_path_or_hf_name : str, 
        lora_adapter_path : str | None = None, 
        debug : bool = False
    ): 
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
            dataset = load_data(tokenizer, split="validation", n_train_tkns=2e4, max_length=256, packing=True)
            loss = compute_loss(model_path_or_hf_name=model_path_or_hf_name, lora_adapter_path=lora_adapter_path, model=model, dataset=dataset)
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

def get_avg_loss(model_name_or_path, lora_adapter_path=None, debug=False, seeds=[1, 2, 3]): 
    avg_loss = 0
    for i in seeds: 
        if lora_adapter_path is not None: 
            _model_name_or_path = model_name_or_path
            _lora_adapter_path = os.path.join(lora_adapter_path, f"seed_{i}", "final_model")
        else: 
            _model_name_or_path = os.path.join(model_name_or_path, f"seed_{i}", "final_model")
            _lora_adapter_path = lora_adapter_path
        
        ckpt_dir = os.path.exists(_model_name_or_path) if lora_adapter_path is None else os.path.exists(_lora_adapter_path)
        if debug and not os.path.exists(ckpt_dir): 
            continue
        loss = get_eval_loss(model_path_or_hf_name=_model_name_or_path, lora_adapter_path=_lora_adapter_path, debug=debug)
        avg_loss += loss
    avg_loss /= len(seeds)
    return avg_loss

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    # Example Usage: 
    # python src/find_opt_lr.py --all_models_path models/pythia-1.4b/packing_n_docs/perturbations/none --save_path configs/optimal_lr_n_docs.json --max_seq_lens 64 128 256
    parser.add_argument("--all_models_path", type=str, default="models/pythia-1.4b/packing/perturbations/none")
    parser.add_argument("--max_seq_lens", type=int, default=[64, 128, 256], nargs="+")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--save_path", type=str, default="configs/optimal_lr.json")
    args = parser.parse_args()
    all_models_path = args.all_models_path
    debug = args.debug
    save_path = args.save_path

    # List comprehension to get all seed model paths 
    model_paths = [root for (root, dirs, files) in os.walk(all_models_path) for dir in dirs if ("seed" in dir) and ("code" not in root) and ("math" not in root)]
    losses = []
    for path in model_paths: 
        if "full" in path: 
            model_name_or_path = path
            lora_adapter_path = None
        else: 
            model_name_or_path = MODEL_DIR_TO_HF_NAME[path.split("/")[1]]
            lora_adapter_path = path
        losses.append(get_avg_loss(model_name_or_path, lora_adapter_path, debug))
    dict_losses = {model_path: loss for model_path, loss in zip(model_paths, losses)}
    # print(dict_losses)
    path_hyperparams = {}
    
    for k, v in dict_losses.items(): 
        dd_in_predefined_list = [dd for dd in DOMAIN_DATASETS if dd in k]
        if len(dd_in_predefined_list) == 0: # dd not in DOMAIN_DATASETS
            continue
        domain_dataset = [dd for dd in DOMAIN_DATASETS if dd in k][0]
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
                "domain_dataset": domain_dataset, 
                "max_seq_len": max_len, 
                "n_tkns": n_tkns, 
                "ft": ft, 
                "lora_rank": r, 
                "lr_loss_map": {
                    lr: v
                }
            }
        else: 
            path_hyperparams[path_wo_lr]["lr_loss_map"][lr] = v

    for model, hyperparams in path_hyperparams.items(): 
        opt_lr, opt_loss = min(hyperparams["lr_loss_map"].items(), key=lambda x: x[1])
        path_hyperparams[model]["opt_lr"] = opt_lr
        path_hyperparams[model]["opt_loss"] = opt_loss
        del path_hyperparams[model]["lr_loss_map"]

    df = pd.DataFrame.from_dict(path_hyperparams, orient='index')
    df.reset_index(inplace=True)
    df.to_json(save_path, orient="records", indent=4)
    exit()

    model_paths_no_seed = [path.split("seed")[0] for path in model_paths]

    # Get last ckpt_dirs from model_paths 
    ckpt_dirs = [get_ckpts(model_path) for model_path in model_paths]

    #get eval loss from ckpt_dirs 
    eval_losses = {
        model_path: get_eval_loss(ckpt_dir) for model_path, ckpt_dir in zip(model_paths, ckpt_dirs)
    }
    
    # compute avg_loss from seeds
    for model_path in model_paths_no_seed: 
        for i in range(1, 4): 
            loss = eval_losses[os.path.join(model_path, f"seed_{i}")]

    get_mean_loss(model_paths_no_seed, eval_losses)
    # [get_eval_loss(ckpt_dir) for ckpt_dir in ckpt_dirs]
    # ckpt_dirs = [[os.path.join(model_path, dir) for dir in os.listdir(model_path) if 'checkpoint' in dir] for model_path in model_paths]
    print(eval_losses)
        # exit()
    exit()
    if not os.path.exists(all_models_path): 
        raise FileNotFoundError(f"Checkpoint path {ft_model_path} does not exist")
    
    # List LR dirs in ckpt_path 
    lr_dirs = [d for d in os.listdir(all_models_path) if os.path.isdir(os.path.join(all_models_path, d))]
    all_models_path = args.all_models_path

    max_seq_lens = args.max_seq_lens
    seeds = [1, 2, 3]
    best_lr_map = {}
    # ".../r_256/lr_2e-4/num_train_all/max_seq_len_128/seed_2/checkpoint-470"
    
    model_size = [s for s in ft_model_path.split("/") if "pythia" in s][0]
    model_details_str = f"Base Model: {model_size}\n"
    if "lora" in ft_model_path: 
        ft_method = "lora"
        rank = ft_model_path.split("r_")[-1]
        model_details_str += f"Finetuning Method: LoRA (r={rank})\n"
    else: 
        ft_method = "full"
        model_details_str += f"Finetuning Method: Full\n"
    
    model_details_str += f"Dataset: {dataset_name}\n"


    for max_seq_len in max_seq_lens: 
        lr_loss_map = {}
        print_str = model_details_str + f"Max Seq Len: {max_seq_len}\n"
        for lr_dir in lr_dirs: 
            avg_loss = 0
            for seed in seeds: 
                ckpt_dir = os.path.join(ft_model_path, lr_dir, f"num_train_{args.num_train}", f"max_seq_len_{max_seq_len}", f"seed_{seed}")
                if not os.path.exists(ckpt_dir): 
                    raise FileNotFoundError(f"Checkpoint dir {ckpt_dir} does not exist")
                
                final_model_path = os.path.join(ckpt_dir, "final_model")
                if not os.path.exists(final_model_path): 
                    raise FileNotFoundError(f"Final model {final_model_path} DOES NOT EXIST! \nREDO FINETUNING")
                
                # List all checkpoint files in ckpt_dir 
                ckpts = [int(f.split("checkpoint-")[-1]) for f in os.listdir(ckpt_dir) if f.startswith("checkpoint")]
                last_ckpt = max(ckpts)

                # Get eval loss from last checkpoint 
                eval_loss_path = os.path.join(ckpt_dir, f"checkpoint-{last_ckpt}", "trainer_state.json")
                with open(eval_loss_path, "r") as f: 
                    eval_loss = json.load(f)
                
                eval_loss = eval_loss["best_metric"]
                avg_loss += eval_loss
            avg_loss /= len(seeds)
            lr_loss_map[lr_dir] = avg_loss
        min_loss_lr = min(lr_loss_map, key=lr_loss_map.get)
        best_lr_map[max_seq_len] = min_loss_lr
        print_str += f"Best LR: {min_loss_lr}\nLoss: {round(lr_loss_map[min_loss_lr], 2)}\n"

        print(print_str)
    # print(best_lr_map)
