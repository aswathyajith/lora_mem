import argparse
from src.utils.interventions import get_full_model_with_low_rank_update
from src.utils.model import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd

BASE_MODEL_NAME_HF_MAP = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-140m": "EleutherAI/pythia-140m",
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_size",
        type=str,
        choices=["pythia-1.4b", "pythia-160m"],
        help="Model size",
        default="pythia-1.4b",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["legal", "code", "math"],
        help="Domains to select from",
        default="code",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        choices=[64, 128, 256],
        help="context lengths (seen by model) to select from",
        default=256,
    )
    parser.add_argument(
        "--n_tkns",
        type=str,
        choices=["2e6"],
        help="No. of input token lengths (seen by model) to select from",
        default="2e6",
    )
    parser.add_argument("--seeds", type=int, default=[1], nargs="+")

    parser.add_argument(
        "--config_path", type=str, default="configs/model_data_train_config.csv"
    )
    parser.add_argument(
        "--target_module",
        type=str,
        default="all-linear",
        choices=["all-linear", "attn_only"],
    )
    parser.add_argument(
        "--ranks", type=int, default=[1, 4, 16, 128, 1024, 2048], nargs="+"
    )
    parser.add_argument(
        "--skip_existing",
        default=False,
        action="store_true",
        help="Skip models that already exist",
    )
    args = parser.parse_args()

    config_path = args.config_path
    target_module = args.target_module
    ranks = args.ranks
    skip_existing = args.skip_existing

    model_configs = pd.read_csv(config_path)
    model_configs = model_configs[
        (model_configs["ft"] == "full")
        & (model_configs["domain"] == args.domain)
        & (model_configs["max_seq_len"] == args.max_seq_len)
        & (model_configs["model_size"] == args.model_size)
    ]
    for _, row in model_configs.iterrows():
        model_size = row["model_size"]
        n_tkns = row["model_path"].split("n_tkns_")[-1].split("/")[0]
        if n_tkns != args.n_tkns:
            continue

        base_model_name = BASE_MODEL_NAME_HF_MAP[model_size]
        tkzr = AutoTokenizer.from_pretrained(base_model_name)
        for rank in ranks:
            for seed in args.seeds:
                model_path = os.path.join(
                    row["model_path"], f"seed_{seed}", "final_model"
                )
                save_path = model_path.replace(
                    "full-ft", f"full-svd/r_{rank}/{target_module}"
                )

                if skip_existing and os.path.exists(save_path):
                    print(f"Skipping {save_path} because it already exists")

                    tkzr.save_pretrained(save_path)
                    continue
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                full_model, tkzr = load_model(
                    model_name=model_path, lora_adapter_path=None
                )
                low_rank_full_model = get_full_model_with_low_rank_update(
                    base_model, full_model, target_module, rank
                )

                # Save the model
                print(f"Saving model to {save_path}")
                low_rank_full_model.save_pretrained(save_path)
                tkzr.save_pretrained(save_path)


if __name__ == "__main__":
    main()
