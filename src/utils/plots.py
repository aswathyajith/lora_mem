import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools

class Plotter:
    def __init__(
            self,
            data_path: str,
            ppl_results_path: str = "configs/optimal_lr.json",
            save_path: str = "test.png"
        ):
        self.data_path = data_path
        self.ppl_results_path = ppl_results_path
        self.save_path = save_path

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

    def plot_losses(self, save_path=None):
        data = self.data
        domains_sorted = data[["domain", "base_ppl"]].groupby("domain").mean().reset_index()
        domains_sorted = domains_sorted.sort_values(by="base_ppl", ascending=False)
        print(domains_sorted)
        domains_sorted = domains_sorted["domain"].tolist()
        print(domains_sorted)
        data["max_seq_len"] = data["max_seq_len"].astype(str)
        data = data.drop(columns=["train_dataset"])
        # Set global font size
        plt.rcParams.update({'font.size': 12})
        
        # Create figure with shared y-axis
        fig, ax = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
        loss_cols = [col for col in data.columns if col.endswith("_ppl")]
        id_vars = [col for col in data.columns if col not in loss_cols]
        
        # Create a single legend for the entire plot
        handles, labels = None, None
        
        for i, domain in enumerate(domains_sorted):
            for j, seq_len in enumerate(data["max_seq_len"].unique()):
                domain_data = data[(data["domain"] == domain) & (data["max_seq_len"] == seq_len)]
                # Melt the data
                melted_data = pd.melt(domain_data, id_vars=id_vars, value_vars=loss_cols)
                plot = sns.barplot(melted_data, x="split", y="value", hue="variable", 
                                 ax=ax[i, j], palette=["#6666ff", "#99cc00", "#ff8000"])
                
                # Store handles and labels from the first plot
                if handles is None:
                    handles, labels = plot.get_legend_handles_labels()
                
                # Remove individual legends if they exist
                legend = ax[i, j].get_legend()
                if legend is not None:
                    legend.remove()
                
                # Add column labels (max_seq_len) at the top
                if i == 0:
                    ax[i, j].set_title(f"Max Sequence Length: {seq_len}", pad=20, fontsize=14)
                
                # Add row labels (domain) on the left
                if j == 0:
                    ax[i, j].set_ylabel(f"Domain: {domain}", rotation=0, labelpad=60, fontsize=14)
                else:
                    ax[i, j].set_ylabel("")
                
                # Remove x-axis labels
                ax[i, j].set_xlabel("")
                
                # Rotate x-axis labels for better readability
                ax[i, j].tick_params(axis='x', labelsize=12)
                ax[i, j].tick_params(axis='y', labelsize=12)
                
                # Ensure y-axis starts at 0
                ax[i, j].set_ylim(bottom=0)
        
        # Add single legend to the figure with more padding
        label_map = {
            "lora_r16_ppl": "Lora (r=16) PPL",
            "base_ppl": "Base PPL",
            "full_ppl": "Full PPL",
        }
        for i in range(len(labels)):
            labels[i] = label_map[labels[i]]
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12, 
                  frameon=False, bbox_to_anchor=(0.5, 0.0))
        
        # Adjust layout to prevent label cutoff with more padding
        # plt.tight_layout(rect=[0, 0.1, 1, 1])
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

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

    def plot_cross_domain_loss(self, df):
        """
        Create nxn barplots of model losses comparing models across domains.
        
        Args:
            data: DataFrame containing the data to plot
            save_path: Optional path to save the figure
        """
        
        save_path = self.save_path
        domains = df.train_domain.unique()
        n_domains = df.train_domain.nunique()
        domain_labels = {domain: domain.capitalize() for domain in domains}
        model_labels = {
            "base_ppl": textwrap.fill("Base", width=10),
            "lora_attn_only_r16_ppl": textwrap.fill("LORA (r=16, Attn Only)", width=10),
            "full_model_ppl": textwrap.fill("Full Model", width=10)
        }

        # Find global min and max values
        y_min = df['loss'].min()
        y_max = df['loss'].max()
        # Add some padding to the limits
        y_range = y_max - y_min
        y_min -= 0.05 * y_range
        y_max += 0.1 * y_range

        # Set up the figure
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(n_domains, n_domains, figsize=(4*n_domains, 4*n_domains))
        
        # Add overall labels
        fig.text(0.5, 0.95, 'Test Domain', ha='center', va='center', fontsize=16)
        fig.text(0.02, 0.5, 'Train Domain', ha='center', va='center', rotation='vertical', fontsize=16)
        
        
        # Create a single legend for the entire plot
        handles, labels = None, None
        
        # Plot each domain combination
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                domain_data = df[(df['train_domain'] == domain1) & (df['test_domain'] == domain2)]
                
                # Create the barplot
                plot = sns.barplot(
                    data=domain_data,
                    x='model',
                    y='loss',
                    hue='model',
                    ax=ax[i, j],
                    palette=["#6666ff", "#99cc00", "#ff8000"]
                )

                # Add value labels on top of each bar
                for container in plot.containers:
                    plot.bar_label(container, fmt='%.2f', padding=3)  # %.2f for 2 decimal places

                # Set the same y-limits for all subplots
                ax[i, j].set_ylim(y_min, y_max)
                
                # Store handles and labels from the first plot
                if handles is None:
                    handles, labels = plot.get_legend_handles_labels()
                
                # Remove individual legends if they exist
                legend = ax[i, j].get_legend()
                if legend is not None:
                    legend.remove()
                
                # Set titles
                if i == 0:
                    ax[i, j].set_title(f"{domain_labels[domain2]}", pad=20, fontsize=14)
                if j == 0:
                    ax[i, j].set_ylabel(f"{domain_labels[domain1]}", rotation=0, labelpad=60, fontsize=14)
                else:
                    ax[i, j].set_ylabel("")
                
                # Rotate x-axis labels
                ax[i, j].tick_params(axis='x')
                
                # Remove x-axis labels for inner plots
                if i != n_domains - 1:
                    ax[i, j].set_xlabel("")

                # convert xticklabels to model_labels
                xticklabels = [model_labels[label.get_text()] for label in ax[i, j].get_xticklabels()]
                ax[i, j].set_xticks(range(len(xticklabels)))
                ax[i, j].set_xticklabels(xticklabels)

                
        
        # Add single legend to the figure
        
        fig.legend(
            handles, 
            labels, 
            loc='lower center', 
            ncol=2, 
            fontsize=12, 
            frameon=False,
            bbox_to_anchor=(0.5, 0.0)
        )
        
        # Adjust layout with more space for labels
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        
        if save_path is not None:
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
