from src.utils.regurgitation import RegurgitatedDataGenerator
import pandas as pd
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--decoding_strategy", type=str, default="greedy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--domain_filter", type=str, default="all")
    parser.add_argument(
        "--path_to_model_data_mapping",
        type=str,
        default="configs/pythia-1.4b-output-configs/model_data_test_config.csv",
    )
    parser.add_argument(
        "--save_dir", type=str, default="data/regurgitation_experiments"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    k = args.k
    batch_size = args.batch_size
    decoding_strategy = args.decoding_strategy
    domain_filter = args.domain_filter
    save_dir = args.save_dir
    # Load config
    config_df = pd.read_csv(args.path_to_model_data_mapping)
    if domain_filter != "all":
        config_df = config_df[config_df["domain"] == domain_filter]

    n_tkns = "2e7"
    exclude_modules = ["attn_only"]
    for _, row in config_df.iterrows():
        row["model_path"] = os.path.join(
            row["model_path"], f"seed_{args.seed}", "final_model"
        )
        # Filter models trained on 2e7 tokens
        if n_tkns not in row["model_path"] or any(
            module in row["model_path"] for module in exclude_modules
        ):
            continue
        if not os.path.exists(row["model_path"]):
            print(f"Model path {row['model_path']} does not exist. Skipping...")
            continue

        else:
            regdata_gen = RegurgitatedDataGenerator(data_config=row)
            regdata_gen.generate_outputs(
                save_dir,
                batch_size=batch_size,
                k=k,
                decoding_strategy=decoding_strategy,
                n=args.sample_size,
            )


if __name__ == "__main__":
    main()
