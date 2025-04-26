import pandas as pd
import json
import os

DOMAINS = ["math", "code", "legal"]
def get_losses_in_file(path_to_ppl_results: str):
    """
    Get the PPLs for each model from the PPL results file
    """
    ppl_results = {}
    with open(path_to_ppl_results, "r") as f:
        ppl_results = json.load(f)
    
    for model, results in ppl_results.items():
        ppl_results[model] = results["eval_loss"]
    return ppl_results

def list_loss_files(path: str):
    """
    List all subdirectories recursively with losses.json in the given path
    """
    loss_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "losses.json":
                loss_files.append(os.path.join(root, file))

    return loss_files

def compute_avg_losses(loss_df: pd.DataFrame):
    """
    Compute the average losses for each model
    """
    ppl_cols = loss_df.columns[loss_df.columns.str.contains("ppl")]

    group_cols = list(set(loss_df.columns) - set(ppl_cols))
    # Get mean and std of ppl_cols
    loss_df[ppl_cols] = loss_df.groupby(group_cols)[ppl_cols].transform("mean")
    loss_df = loss_df.drop_duplicates(subset=group_cols)
    return loss_df

def get_all_losses(path: str):
    """
    Get list of all losses.json files and extracts the losses for each model
    """
    losses = {}
    for domain in DOMAINS:
        loss_files = list_loss_files(os.path.join(path, domain))
        
        for loss_file in loss_files:
            losses[loss_file] = get_losses_in_file(loss_file)
            seed = loss_file.split("seed_")[-1].split("/")[0]
            ds_fields = loss_file.split("model_outputs/")[-1].split("/")
            domain = ds_fields[0]
            train_dataset = ds_fields[1]
            eval_dataset = ds_fields[2]
            split = "train" if "train" in loss_file else "val"
            max_seq_len = loss_file.split("max_seq_len_")[-1].split("/")[0]
            n_tkns = loss_file.split("n_tkns_")[-1].split("/")[0]
            losses[loss_file]["seed"] = seed
            losses[loss_file]["domain"] = domain
            losses[loss_file]["train_dataset"] = train_dataset
            losses[loss_file]["eval_dataset"] = eval_dataset
            losses[loss_file]["max_seq_len"] = max_seq_len
            losses[loss_file]["split"] = "pretrain" if eval_dataset=="pretraining" else split 
            losses[loss_file]["n_tkns"] = n_tkns
        loss_df = pd.DataFrame.from_dict(losses, orient="index")
        loss_df = loss_df.reset_index(drop=True)
        loss_df = loss_df[loss_df["n_tkns"] == "2e6"]
        # Re-order columns
        loss_cols = list(loss_df.columns[loss_df.columns.str.contains("ppl")])
        loss_cols.sort()
        print(loss_cols)
        loss_df["max_seq_len"] = loss_df["max_seq_len"].astype(int)
        loss_df = loss_df[["domain", "train_dataset", "split", "eval_dataset", "max_seq_len"] + loss_cols]
        loss_df = compute_avg_losses(loss_df)
    return loss_df

def save_latex_table(loss_df: pd.DataFrame, latex_path = "results/latex_tables.txt"):
    # Convert to latex table with multicolumns merged]
    # loss_df.columns = [col.replace("_", " ") for col in loss_df.columns]
    loss_cols = list(loss_df.columns[loss_df.columns.str.contains("ppl")])
    index = [col for col in loss_df.columns if col not in loss_cols]
    for col in loss_cols:
        loss_df[col] = loss_df[col].apply(lambda x: '%.3f' % x)
    loss_df = loss_df.set_index(index)
    # Sort by index
    loss_df = loss_df.sort_index()

    latex_str = loss_df.to_latex(
        index=True, 
        multirow=True,
        escape=False, 
        float_format=lambda x: '%.3f' % x,  # Format numbers
        caption="Domain-wise losses across different datasets",
        # formatters={col: lambda x: '%.3f' % x for col in loss_cols}  # Format string
        )
    
    latex_str = latex_str.replace("[t]", "").replace("_", "\_").replace("_ppl", "")

    with open(latex_path, "w") as f:
        f.write(latex_str)

def compute_agg_losses(df: pd.DataFrame):
    """
    Compute the average losses for each model
    """
    # aggregate over domain, train_dataset, split, max_seq_len to get mean losses
    copy = df.drop(columns=["eval_dataset"])
    loss_cols = list(copy.columns[copy.columns.str.contains("ppl")])
    # Convert to float
    copy[loss_cols] = copy[loss_cols].astype(float)
    copy = copy.groupby(["domain", "train_dataset", "split", "max_seq_len"]).mean().reset_index()
    return copy

if __name__ == "__main__":
    loss_df = get_all_losses("data/model_outputs")
    print(loss_df.columns)
    save_latex_table(loss_df)
    agg_loss_df = compute_agg_losses(loss_df)


# Save agg_loss_df
    agg_loss_df.to_csv("results/agg_losses.csv", index=False)
