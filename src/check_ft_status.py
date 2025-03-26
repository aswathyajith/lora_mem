import os 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_models", type=str, default=None)
    parser.add_argument("--data_preprocess", type=str, default="packing")
    parser.add_argument("--model_sizes", type=str, default=["pythia-1.4b"], nargs="+")
    # parser.add_argument("--dataset_domains", type=str, default=["biomed/chemprot", "wiki/wikitext", "bible/bible_corpus_eng"], nargs="+")
    dataset_domains = [
        # "wiki/wikitext",
        "biomed/chemprot",
        # "bible/bible_corpus_eng"
    ]
    parser.add_argument("--dataset_domains", type=str, default=dataset_domains, nargs="+")
    parser.add_argument("--full_ft_lrs", type=str, default=["2e-5", "2e-6", "2e-7"], nargs="+")
    parser.add_argument("--lora_ranks", type=int, default=[256], nargs="+")
    parser.add_argument("--lora_lrs", type=str, default=["2e-2", "2e-3", "2e-4"], nargs="+")
    parser.add_argument("--num_trains", type=int, default=["all"], nargs="+")
    parser.add_argument("--max_seq_lens", type=int, default=[256], nargs="+")
    parser.add_argument("--seeds", type=int, default=[1, 2, 3], nargs="+")
    args = parser.parse_args()

    model_sizes = args.model_sizes
    dataset_domains = args.dataset_domains
    full_ft_lrs = args.full_ft_lrs
    num_trains = args.num_trains
    max_seq_lens = args.max_seq_lens
    seeds = args.seeds
    lora_ranks = args.lora_ranks
    lora_lrs = args.lora_lrs
    path_to_models = args.path_to_models
    data_preprocess = args.data_preprocess

    missing_models = []
    n = 0
    for model in model_sizes:
        for dataset_domain in dataset_domains:
            for full_ft_lr in full_ft_lrs:
                for num_train in num_trains:
                    for max_seq_len in max_seq_lens:
                        
                        # for seed in seeds:
                        #     path_to_model = f"models/{model}/{data_preprocess}/{dataset_domain}/full-ft/lr_{full_ft_lr}/num_train_{num_train}/max_seq_len_{max_seq_len}/seed_{seed}/final_model"
                        #     if not os.path.exists(path_to_model):
                        #         missing_models.append(path_to_model)    
                        #     n += 1
                        
                        for lora_rank in lora_ranks:
                            for lora_lr in lora_lrs:
                                for seed in seeds:
                                    path_to_model = f"models/{model}/{data_preprocess}/{dataset_domain}/lora/r_{lora_rank}/lr_{lora_lr}/num_train_{num_train}/max_seq_len_{max_seq_len}/seed_{seed}/final_model"
                                    if not os.path.exists(path_to_model):
                                        missing_models.append(path_to_model)
                                    n += 1

    print(missing_models)
    print(f"Total missing models: {len(missing_models)} / {n}")
    
