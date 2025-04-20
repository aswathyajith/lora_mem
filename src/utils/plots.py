import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(
            self,
            data_path: str,
            ppl_results_path: str = "configs/optimal_lr.json",
        ):
        self.data_path = data_path
        self.ppl_results_path = ppl_results_path

    def read_data(self):
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r") as f:
                self.data = pd.read_json(f, orient="records", lines=True)
        elif self.data_path.endswith(".csv"):
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {self.data_path}")

    def process_data(self):
        data = self.data
        prob_cols = [col for col in data.columns if col.endswith("_prob")]
        prob_cols.sort()
        lora_cols = [col for col in prob_cols if "lora" in col]
        for col in lora_cols:
            data[f"full_minus_{col.replace('_prob', '')}"] = data["full_prob"] - data[col]
            data[f"{col.replace('_prob', '')}_minus_base"] = data[col] - data["base_prob"]
        data["full_minus_base"] = data["full_prob"] - data["base_prob"]
        self.data = data
        return data

    def plot_prob_diffs(self, kde=False, save_path=None):
        data = self.data
        if kde:
            sns.kdeplot(data["full_minus_base"], label="Full - Base")
            sns.kdeplot(data["full_minus_lora_r1"], label="Full - Lora r1")
            sns.kdeplot(data["full_minus_lora_r2"], label="Full - Lora r2")
            sns.kdeplot(data["full_minus_lora_r3"], label="Full - Lora r3")
        else:
            sns.scatterplot(data, x="full_minus_base", y="lora_r16_minus_base", hue="base_prob", s=5)
            # Add labels
            plt.xlabel("Full - Base")
            plt.ylabel("Lora r16 - Base")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
        plt.legend()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

plotter = Plotter("docs/data/code/starcoder/apps/test/n_tkns_2e6/max_seq_len_64/seed_1/combined_results_10000.json")
plotter.read_data()
plotter.process_data()
plotter.plot_prob_diffs(save_path="docs/data/code/starcoder/apps/test/n_tkns_2e6/max_seq_len_64/seed_1/combined_results_10000.png")
plt.show()