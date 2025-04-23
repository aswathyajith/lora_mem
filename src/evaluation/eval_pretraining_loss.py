import os
from src.utils.data import load_pretraining_data
from src.utils.model import load_model, compute_loss
import argparse
import json
from datasets import load_from_disk, Dataset
import pandas as pd
import shutil

base_model_map = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b", 
    "pythia-160m": "EleutherAI/pythia-160m", 
}

def get_pretraining_data(tokenizer, max_seq_len, n_tkns, save_path):
    """
    Get the pretraining data for the given tokenizer.
    """
    ds = load_pretraining_data(max_seq_length=max_seq_len, n_tkns=n_tkns)   
    # decode "text" field from "input_ids"
    
    ds = ds.map(lambda x: {"text": tokenizer.batch_decode(x["input_ids"])}, batched=True)
    print(ds)
    print(ds["text"][0])
    # Save the dataset
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving pretraining data to {save_path}")
    ds.save_to_disk(save_path)
    return ds

def get_loss(
        model_path: str, 
        ds: Dataset, 
        base_model_name: str,
        ft: str | None = None, 
        max_seq_len: int = 64, 
        text_field: str = "text"
    ): 
    """
    Get the loss of the model on the pretraining data.
    """

    # base model eval
    if ft is None:
        model, tokenizer = load_model(base_model_name, lora_adapter_path=None)
        loss = compute_loss(
            model_path_or_hf_name=base_model_name,
            model=model,
            tokenizer=tokenizer, 
            dataset=ds, 
            text_field=text_field, 
            max_seq_length=max_seq_len, 
            packing=True, 
            base_model=True
        )

    else:
        if ft == "full": # load full model
            model, tokenizer = load_model(model_path, lora_adapter_path=None)
        else: # load lora model
            model, tokenizer = load_model(base_model_name, lora_adapter_path=model_path)

        loss = compute_loss(
            model_path_or_hf_name=model_path, 
            model=model, 
            tokenizer=tokenizer, 
            dataset=ds, 
            text_field=text_field, 
            max_seq_length=max_seq_len, 
            packing=True, 
            base_model=False
        )
    
    return loss

def get_path_to_pretraining_data(model_name, max_seq_len, save_dir):
    save_path = os.path.join(save_dir, f"{model_name}", f"max_seq_len_{max_seq_len}", f"tokenized_samples.hf")
    return save_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--n_tkns_to_save", type=int, default=2e5)
    parser.add_argument("--data_save_dir", type=str, default="data/pretraining_data/the_pile")
    parser.add_argument("--model_selection_path", type=str, default="configs/model_data_train_config.csv")
    parser.add_argument("--seeds", type=int, default=[1, 2, 3], nargs="+")
    args = parser.parse_args()

    models_to_eval = pd.read_csv(args.model_selection_path)
    seeds = args.seeds
    models_to_eval = models_to_eval[["model_path", "domain", "dataset_name", "model_size", "ft", "lora_rank", "max_seq_len"]].drop_duplicates()
    print(f"Finetuned models to evaluate: {len(models_to_eval)}")
    
    for _, row in models_to_eval.iterrows():
        model_size = row["model_size"]
        base_model_name = base_model_map[model_size]
        max_seq_len = row["max_seq_len"]

        # Download pretraining data if it doesn't exist
        data_path = get_path_to_pretraining_data(model_size, max_seq_len, args.data_save_dir)
        if not os.path.exists(data_path):
            _, tokenizer = load_model(base_model_name, lora_adapter_path=None)
            ds = get_pretraining_data(tokenizer, max_seq_len, n_tkns=args.n_tkns_to_save, save_path=data_path)
        else:
            print(f"Loading pretraining data at {data_path}")
            ds = load_from_disk(data_path)

        model_path = row["model_path"]
        domain = row["domain"]
        dataset_name = row["dataset_name"]
        n_tkns = model_path.split("n_tkns_")[-1].split("/")[0]
        ft = row["ft"]
        lora_rank = row["lora_rank"]

        # Get output directory
        output_dir_root = os.path.join("data/model_outputs", domain, dataset_name, "pretraining/train", f"n_tkns_{n_tkns}", f"max_seq_len_{max_seq_len}")

        losses = {}

        # Check if base model eval is already done
        base_output_path = os.path.join("data/model_outputs/base_model_outputs", f"{model_size}", "pretraining", "train", f"max_seq_len_{max_seq_len}", "losses.json")
        print(f"base_output_path: {base_output_path}")
        if os.path.exists(base_output_path):
            print(f"Base model eval already done for {model_path}")
            base_loss = json.load(open(base_output_path, "r"))["base_ppl"]
        else:
            os.makedirs(os.path.dirname(base_output_path), exist_ok=True)
            # Base model eval
            base_loss = get_loss(
                model_path=model_path, 
                ds=ds, 
                base_model_name=base_model_name, 
                ft=None, 
                max_seq_len=max_seq_len, 
                text_field="text"
            )
            # Save base model loss
            with open(base_output_path, "w") as f:
                json.dump({"base_ppl": base_loss}, f)


        # Check if finetuned model eval is already done
        ft_key = f"{ft}_r{int(lora_rank)}_ppl" if ft != "full" else f"{ft}_ppl"

        for seed in seeds:
            output_path = os.path.join(output_dir_root, f"seed_{seed}", "losses.json")
            loss = None
            if os.path.exists(output_path): 
                # load losses from output_path
                
                with open(output_path, "r") as f:
                    losses = json.load(f)
                    loss = losses.get(ft_key, None)
                
            if loss is not None:
                print(f"Finetuned model eval already done for {model_path}")

            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                losses[ft_key] = get_loss(
                    model_path=os.path.join(model_path, f"seed_{seed}", "final_model"), 
                    ds=ds, 
                    base_model_name=base_model_name, 
                    ft=ft, 
                    max_seq_len=max_seq_len, 
                    text_field="text"
                )

            # Save loss
            losses["base_ppl"] = base_loss
            with open(output_path, "w") as f:
                json.dump(losses, f)

        print("Losses:\n", losses)
        print(f"\nSaved losses to {output_dir_root}")

if __name__ == "__main__":
    main()