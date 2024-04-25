from transformers import AutoTokenizer
import torch
import pandas as pd
import argparse
import os

class MemChecker:
    def __init__(self, model_comps_path):

        # Initialize tokenizer
        self.model_comps_path = model_comps_path

        # Load model completion tensors from disk 
        model_comps = torch.load(model_comps_path)
        self.model_comps = model_comps
        
    def suffix_div(self, context_len): 
        def get_first_div_pos(a, b): 
            # gets the index of the first pos at which two tensors diverge
            disagreement_mask = (a[:,context_len:] != b[:,context_len:]).long()
            return torch.argmax(disagreement_mask, dim=1)

        
        model_comps=self.model_comps
        orig_seq_ids, model_comp_ids = torch.split(model_comps, split_size_or_sections=1, dim=1)
        orig_seq_ids = torch.squeeze(orig_seq_ids, dim=1)
        model_comp_ids = torch.squeeze(model_comp_ids, dim=1)
        # print(orig_seq_ids, model_comp_ids)
        divergence_pos = get_first_div_pos(orig_seq_ids, model_comp_ids)
        
        return divergence_pos
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--output_file_path", default="/net/projects/clab/aswathy/projects/lora_mem/results/pythia-160m/pretrained/epoch_20/train_set_completions/len_25/inp_out_tensors.pt", type=str)
    # parser.add_argument("--prompt_context_len", default=25, type=int)
    # parser.add_argument("--predefined_context_lens", action=argparse.BooleanOptionalAction)
    
    # args = parser.parse_args()

    models = ["pythia-160m", "pythia-410m", "pythia-1.4b"]
    learning_setups = ["pretrained", "lora-ft", "full-ft"]

    models = ["pythia-1.4b"]
    learning_setups = ["full-ft"]

    splits = ["train", "val"]
    context_lens = [25, 50, 75]

    # models = ["pythia-160m"]
    # learning_setups = ["pretrained"]
    # splits = ["train"]
    # context_lens = [25]

    df = pd.DataFrame(columns=["model", "learning_setup", "split", "prefix_context_len", "n_seq", "avg_len_to_diverge", "n_seq_with_min_1_tkn_match", "avg_len_to_diverge_with_min_1_tkn_match"])

    for model in models:
        for setup in learning_setups: 
            for split in splits: 
                for c_len in context_lens: 
                    output_path = os.path.join("results", model, setup, "epoch_20", f"{split}_set_completions", f"len_{c_len}", "inp_out_tensors.pt")
                    if os.path.exists(output_path): 
                        mem_ch = MemChecker(output_path)
                        div_pos = mem_ch.suffix_div(c_len)

                        avg_div_pos_tot = torch.mean(div_pos.float()).item()
                        n_tot = div_pos.shape[0]
                        # print(f"Average length at which output diverges from original sequence for the {n_tot} sequences with length > {c_len}: {avg_div_pos_tot}")
                        
                        nonzero_div = torch.squeeze(div_pos[div_pos.nonzero()], 1)
                        avg_div_pos_1_match = torch.mean(nonzero_div.float()).item()
                        n_1_match = nonzero_div.shape[0]
                        # print(f"Average length at which output diverges from original sequence (when at least 1 token matches) for the {n} sequences with length > {prompt_context_len}: {avg_div_pos}")
                        
                        new_row = {
                            "model": model, 
                            "learning_setup": setup, 
                            "split": split, 
                            "prefix_context_len": c_len, 
                            "n_seq": n_tot,
                            "avg_len_to_diverge": round(avg_div_pos_tot, 2), 
                            "n_seq_with_min_1_tkn_match": n_1_match, 
                            "avg_len_to_diverge_with_min_1_tkn_match": round(avg_div_pos_1_match, 2)
                        }

                        df.loc[len(df)] = new_row
    df.to_csv("results/stats.csv", index=False)

    # output_file_path = args.output_file_path
    # prompt_context_len = args.prompt_context_len

    # predefined_context_lens = args.predefined_context_lens



    