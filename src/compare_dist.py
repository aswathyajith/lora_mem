### Compare output distributions of different models

import numpy as np
import pandas as pd
from pandas import DataFrame
import itertools
import re
from ast import literal_eval

class PairwiseDistComp:
    def __init__(self, 
                 dataset: DataFrame, 
                 dist_cols: list):
        self.dataset = dataset
        self.dist_cols = dist_cols

    def _iou(self, 
            row, 
            col1: pd.Series, 
            col2: pd.Series, 
            k: int = 5):
        '''
        Compute the Intersection Over Union (IoU) of 
        the output token dist over 2 columns/models 
        for the top k elements from each list.
        '''
        
        set1 = set(row[col1][:k])
        set2 = set(row[col2][:k])
        return len(set1 & set2) / len(set1 | set2)
    

    def get_dist_diff(self, 
                     col1: str, 
                     col2: str,
                     k: int = 5,
                     new_col_name: str = None):
        '''
        Compare different output distributions in the dataset.
        '''
        
        # Get unordered pairs of dist_cols
        self.dataset[new_col_name] = self.dataset.apply(
            lambda row: self._iou(row, col1, col2, k), axis=1)
        
    def compare_model_pairs(self, topk):
        pairs = list(itertools.combinations(self.dist_cols, 2))

        # Apply IoU to each pair of columns
        for c1, c2 in pairs:
            new_col_name = f"{c1.replace('_top_k_pred_tokens', '')}_{c2.replace('_top_k_pred_tokens', '')}_iou"
            
            self.get_dist_diff(c1, c2, topk, new_col_name)

        return self.dataset


if __name__ == "__main__":
    data_path = "data/plotting_data/pythia-1.4b/packing_n_docs/perturbations/none/bible/bible_corpus_eng/train/num_train_all/max_seq_len_128/sample_2048/seed_1/lora_r16_full_merged.json"
    dataset = pd.read_json(data_path, orient="records")

    # Parsing column values from string to list
    def replace_ws(s): 
        s = re.sub(r'\[\s+', '[', s)
        return re.sub(r'(?<!\[)\s+(?!\])', ',', s)
    for col in dataset.columns:
        if "_top_k_pred_tokens" in col or "_top_k_pred_probs" in col:
            dataset[col] = dataset[col].apply(replace_ws).apply(literal_eval)

    cols = [col for col in dataset.columns if "_top_k_pred_tokens" in col]
    dist_comp = PairwiseDistComp(dataset, cols)

    # exit()
    df = dist_comp.compare_model_pairs(topk=5)
    save_path = data_path.replace(".json", "_iou.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")