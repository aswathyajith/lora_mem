from src.utils.generator import TokenGenerator
from src.utils.data import load_data
from datasets import Dataset
import argparse
import pandas as pd
import os
import shutil
from src.utils.model import compute_loss
import json
import torch

base_model_map = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-160m": "EleutherAI/pythia-160m",
}


def save_base_model_outputs(
    base_model_name: str,
    ds: Dataset,
    base_output_path: str,
    base_vocab_dist_path: str,
    n_output_samples: int = 100,
):
    """
    Compute and save base model outputs to a base_output_path json dataframe
    """
    os.makedirs(os.path.dirname(base_output_path), exist_ok=True)
    base_tkn_gen = TokenGenerator(base_model_name, lora_adapter_path=None)
    sampled_ds = base_tkn_gen.get_n_output_samples(ds, n_output_samples)
    try:
        base_model_outputs, vocab_dist = base_tkn_gen.iterate_over_ds(sampled_ds)
        base_model_outputs = pd.DataFrame(base_model_outputs)
        base_model_outputs.to_json(base_output_path, orient="records", lines=True)
        torch.save(vocab_dist, base_vocab_dist_path)
    except Exception as e:
        print(f"Error getting base model outputs for {base_model_name}: {e}")

    return base_model_outputs


def filter_configs(model_data_mapping: pd.DataFrame, domains: list[str]):
    model_data_mapping = model_data_mapping[model_data_mapping["domain"].isin(domains)]
    return model_data_mapping


def map_config_to_full_svd_configs(
    config_df: pd.DataFrame, ranks: list[int], target_modules: list[str]
) -> pd.DataFrame:
    """
    Map full model configs to full-svd configs and returns the config dataframe.

    Args:
        model_data_config_path: Path to the model data config file.
        ranks: List of ranks to map to.
        target_modules: List of target modules to map to.

    Returns:
        pd.DataFrame: A dataframe of the full-svd configs.

    Example Usage: map_config_to_full_svd_configs(configs/model_data_train_config.csv, [1, 4], ["all-linear", "attn-attn_only"])

    This function creates a new config dataframe to generate combinations of ranks and target modules.
    """

    full_models = config_df[config_df["ft"] == "full"]
    svd_conf = []

    for i, row in full_models.iterrows():
        model_path = row["model_path"]

        for rank in ranks:
            for target_module in target_modules:
                svd_path = model_path.replace(
                    "full-ft", f"full-svd/r_{rank}/{target_module}"
                )
                svd_row = row.copy(deep=True)
                svd_row["model_path"] = svd_path
                svd_row["model_key_ppl"] = f"full_svd_r{rank}_{target_module}_ppl"
                svd_row["ft"] = "full-svd"
                svd_row["ft_str"] = f"full-svd/r_{rank}/{target_module}"
                svd_conf.append(svd_row)

    df = pd.DataFrame.from_records(svd_conf)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_model_data_mapping",
        type=str,
        default="configs/model_data_train_config.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/generations",
        help="Directory to save model outputs (model outputs, vocab dist)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/ppl",
        help="Directory to save model results (PPL)",
    )
    parser.add_argument(
        "--base_model_output_dir",
        type=str,
        default="data/base_outputs_cached",
        help="Directory to save base model outputs",
    )
    parser.add_argument(
        "--base_model_results_dir",
        type=str,
        default="results/base_ppl_cached",
        help="Directory to save base model results (PPL)",
    )
    parser.add_argument(
        "--n_output_samples",
        type=int,
        default=100,
        help="Number of output samples to generate for analysis",
    )
    parser.add_argument(
        "--domains", type=str, nargs="+", default=["code", "math", "legal"]
    )
    parser.add_argument("--n_tkns", type=str, nargs="+", default=["2e6"])
    parser.add_argument("--max_seq_lens", type=int, nargs="+", default=[256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1])
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--cross_domain", action="store_true", default=False)
    parser.add_argument("--full_svd_models", default=False, action="store_true")
    parser.add_argument("--in_distribution", action="store_true", default=False)
    parser.add_argument(
        "--svd_target_modules", type=str, nargs="+", default=["all-linear"]
    )
    parser.add_argument("--svd_ranks", type=int, nargs="+", default=[128])
    args = parser.parse_args()

    model_output_dir_root = args.output_dir
    base_model_output_dir_root = args.base_model_output_dir
    model_results_dir_root = args.results_dir
    base_model_results_dir_root = args.base_model_results_dir
    eval_only = args.eval_only
    cross_domain = args.cross_domain
    n_output_samples = args.n_output_samples

    if args.full_svd_models:
        model_data_mapping = map_config_to_full_svd_configs(
            pd.read_csv(args.path_to_model_data_mapping),
            args.svd_ranks,
            args.svd_target_modules,
        )
    else:
        model_data_mapping = pd.read_csv(args.path_to_model_data_mapping)
    eval_split = (
        "train" if "train" in args.path_to_model_data_mapping.split("/")[-1] else "test"
    )

    for base_model in base_model_map:
        config_by_base_model = model_data_mapping[
            model_data_mapping["model_size"] == base_model
        ]
        if args.domains is not None:
            config_by_base_model = filter_configs(config_by_base_model, args.domains)
        tokenizer = TokenGenerator(
            base_model_map[base_model], lora_adapter_path=None
        ).tokenizer
        print(len(config_by_base_model))
        for _, row in config_by_base_model.iterrows():
            model_path_root = row["model_path"]

            if not os.path.exists(model_path_root):
                print(f"Model path {model_path_root} does not exist. Skipping...")
                continue

            ft = row["ft"]
            base_model = row["model_size"]
            base_model_name = base_model_map[base_model]
            domain = row["domain"]
            dataset = row["dataset_name"]
            max_seq_len = row["max_seq_len"]

            model_n_tkns = model_path_root.split("n_tkns_")[-1].split("/")[0]
            if (
                (model_n_tkns not in args.n_tkns)
                or (max_seq_len not in args.max_seq_lens)
                or (ft == "lora" and row["lora_rank"] not in args.ranks)
            ):
                print(
                    f"n_tkns is not in {args.n_tkns} or max_seq_len is not {args.max_seq_lens} or lora_rank is not in {args.ranks}. Skipping..."
                )
                continue

            if args.in_distribution:
                if dataset not in model_path_root:
                    print(f"{dataset} out of domain for {model_path_root}. Skipping...")
                    continue

            print("Evaluating model at: ", model_path_root)
            # Load data
            ds = load_data(tokenizer, max_length=max_seq_len, **row)

            base_output_path = os.path.join(
                base_model_output_dir_root,
                base_model,
                domain,
                dataset,
                eval_split,
                f"max_seq_len_{max_seq_len}",
                "base_model_outputs.json",
            )
            base_vocab_dist_path = os.path.join(
                base_model_output_dir_root,
                base_model,
                domain,
                dataset,
                eval_split,
                f"max_seq_len_{max_seq_len}",
                "vocab_dist_base_model_outputs.pt",
            )
            base_ppl_path = os.path.join(
                base_model_results_dir_root,
                base_model,
                domain,
                dataset,
                eval_split,
                f"max_seq_len_{max_seq_len}",
                "losses.json",
            )

            # Get base model outputs and save if it doesn't exist
            if not eval_only and not os.path.exists(base_output_path):
                save_base_model_outputs(
                    base_model_name,
                    ds,
                    base_output_path,
                    base_vocab_dist_path,
                    n_output_samples,
                )

            if not os.path.exists(base_ppl_path):
                ppl_args = {
                    "model_path_or_hf_name": base_model_name,
                    "lora_adapter_path": None,
                }
                ppl = compute_loss(
                    **ppl_args,
                    dataset=ds,
                    text_field=row["text_field"],
                    max_seq_length=max_seq_len,
                    base_model=True,
                )
                ppl_dict = {"base_ppl": ppl}
                os.makedirs(os.path.dirname(base_ppl_path), exist_ok=True)
                with open(base_ppl_path, "w") as f:
                    json.dump(ppl_dict, f)

            for seed in args.seeds:
                model_path = os.path.join(
                    model_path_root, f"seed_{seed}", "final_model"
                )
                if not os.path.exists(model_path):
                    print(f"Model path {model_path} does not exist. Skipping...")
                    continue
                model_dataset = model_path.split("/" + domain + "/")[-1].split("/")[0]

                model_n_tkns = model_path.split("n_tkns_")[-1].split("/")[0]
                if cross_domain:
                    trained_on_domain = model_path.split("/")[5]
                    trained_on_dataset = model_path.split("/")[6]
                    tested_on_domain = domain
                    tested_on_dataset = dataset
                    sub_dir_data = os.path.join(
                        "cross_domain",
                        trained_on_domain,
                        trained_on_dataset,
                        tested_on_domain,
                        tested_on_dataset,
                        f"n_tkns_{model_n_tkns}",
                        f"max_seq_len_{max_seq_len}",
                        f"seed_{seed}",
                    )
                else:
                    sub_dir_data = os.path.join(
                        domain,
                        model_dataset,
                        dataset,
                        model_n_tkns,
                        f"max_seq_len_{max_seq_len}",
                        f"seed_{seed}",
                        eval_split,
                    )

                model_output_dir = os.path.join(
                    model_output_dir_root, base_model, sub_dir_data
                )
                model_results_dir = os.path.join(
                    model_results_dir_root, base_model, sub_dir_data
                )

                if not eval_only:
                    print("OUTPUT DIR", model_output_dir)
                    os.makedirs(model_output_dir, exist_ok=True)

                print("MODEL RESULTS DIR", model_results_dir)
                os.makedirs(model_results_dir, exist_ok=True)

                if ft == "full" or ft == "full-svd":
                    if not eval_only:
                        ft_str = "full" if ft == "full" else row["ft_str"]
                        filename = f"{ft_str}_model_outputs.json"
                        tkn_gen_args = {
                            "model_name": model_path,
                            "lora_adapter_path": None,
                        }

                    ppl_args = {
                        "model_path_or_hf_name": model_path,
                        "lora_adapter_path": None,
                    }
                    model_loss_name = (
                        "full_ppl" if ft == "full" else row["model_key_ppl"]
                    )

                else:
                    module = model_path.split("/lora/")[1].split("/")[0]
                    if not eval_only:
                        filename = (
                            f"lora_{module}_r{int(row['lora_rank'])}_model_outputs.json"
                        )
                        tkn_gen_args = {
                            "model_name": base_model_name,
                            "lora_adapter_path": model_path,
                        }

                    ppl_args = {
                        "model_path_or_hf_name": base_model_name,
                        "lora_adapter_path": model_path,
                    }
                    model_loss_name = f"lora_{module}_r{int(row['lora_rank'])}_ppl"

                path_to_ppls = os.path.join(model_results_dir, "losses.json")
                loss_computed = False
                ppl_dict = {}
                if os.path.exists(path_to_ppls):
                    with open(path_to_ppls, "r") as f:
                        ppl_dict = json.load(f)
                        if model_loss_name in ppl_dict:
                            print(
                                f"Loss already exists for {model_loss_name}. Skipping..."
                            )
                            loss_computed = True

                if not loss_computed:
                    ppl = compute_loss(
                        **ppl_args,
                        dataset=ds,
                        text_field=row["text_field"],
                        max_seq_length=max_seq_len,
                    )

                    ppl_dict[model_loss_name] = ppl

                if "base_ppl" not in ppl_dict:
                    # open base ppl
                    with open(base_ppl_path, "r") as f:
                        base_ppl_dict = json.load(f)
                        base_ppl = base_ppl_dict["base_ppl"]
                        ppl_dict["base_ppl"] = base_ppl

                with open(path_to_ppls, "w") as f:
                    json.dump(ppl_dict, f)

                # Get model outputs
                if not eval_only:
                    # Copy base model outputs to output dir
                    base_path = os.path.join(
                        model_output_dir, "base_model_outputs.json"
                    )
                    if not os.path.exists(base_path):
                        shutil.copy(base_output_path, model_output_dir)
                        shutil.copy(base_vocab_dist_path, model_output_dir)

                    model_outputs_path = os.path.join(model_output_dir, filename)
                    vocab_dist_path = os.path.join(
                        model_output_dir, "vocab_dist_" + filename
                    ).replace(".json", ".pt")
                    if os.path.exists(model_outputs_path):
                        print(
                            f"Model outputs already exist for {model_outputs_path}. Skipping..."
                        )
                        continue
                    tkn_gen = TokenGenerator(**tkn_gen_args)
                    sampled_ds = tkn_gen.get_n_output_samples(ds, n_output_samples)
                    # Generate model outputs
                    model_outputs, vocab_dist = tkn_gen.iterate_over_ds(sampled_ds)
                    model_outputs = pd.DataFrame(model_outputs)

                    # Save model outputs
                    model_outputs.to_json(
                        model_outputs_path, orient="records", lines=True
                    )

                    # Save vocab dist
                    torch.save(vocab_dist, vocab_dist_path)


if __name__ == "__main__":
    main()
