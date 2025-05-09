import pandas as pd
import os
from transformers import AutoTokenizer

def prep_data_for_viz(base_model: str = "EleutherAI/pythia-1.4b", path_to_model_outputs: str = "data/model_outputs/math/open_web_math/open_web_math/test/n_tkns_2e6/max_seq_len_256/seed_1"): 
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    outputs_paths = [p for p in os.listdir(path_to_model_outputs) if p.endswith("model_outputs.json")]
    all_dfs = None
    merge_keys = ["seq_id", "context_len", "curr_token"]
    for path in outputs_paths: 
        print(os.path.join(path_to_model_outputs, path))
        prefix = path.split("_model_outputs")[0]
        df = pd.read_json(os.path.join(path_to_model_outputs, path), orient="records", lines=True)
        df = df[["seq_id", "context_len", "curr_token", "norm_probs", "curr_token_rank", "top_k_pred_tokens", "top_k_pred_probs", "entropy"]]
        df.rename(columns={"norm_probs": "prob"}, inplace=True)
        
        # Decode top_k_pred_tokens
        
        df["top_k_pred_tokens"] = df["top_k_pred_tokens"].apply(lambda x: tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True))
        df = df.rename(columns={
            old: new for old, new in zip([col for col in df.columns if col not in merge_keys], [f"{prefix}_{col}" for col in df.columns if col not in merge_keys])
        })
        print(df.columns)
        if all_dfs is None:
            all_dfs = df
        else:
            all_dfs = all_dfs.merge(df, on=merge_keys)

    all_dfs.to_json(os.path.join(path_to_model_outputs, "combined_outputs.json"), orient="records", lines=True)

prep_data_for_viz(path_to_model_outputs="data/model_outputs/math/open_web_math/open_web_math/test/n_tkns_2e6/max_seq_len_128/seed_1")
prep_data_for_viz(path_to_model_outputs="data/model_outputs/legal/us_bills/us_bills/test/n_tkns_2e6/max_seq_len_128/seed_1")
prep_data_for_viz(path_to_model_outputs="data/model_outputs/code/starcoder/starcoder/test/n_tkns_2e6/max_seq_len_128/seed_1")
