import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def compute_topk_overlap(p: torch.Tensor, q: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute the top-k overlap between corresponding rows of two probability distributions.
    """
    return torch.sum(
        torch.topk(p, k, dim=1).indices == torch.topk(q, k, dim=1).indices,
        dim=1,
        keepdim=True,
    )


def compute_js_distance(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute the Jensen-Shannon distance between corresponding rows of two probability distributions.

    Args:
        p (torch.Tensor): First probability distribution tensor of shape (N, V)
        q (torch.Tensor): Second probability distribution tensor of shape (N, V)
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: JS distances of shape (N, 1)
    """
    # Input validation
    if torch.any(p < 0) or torch.any(q < 0):
        raise ValueError("Input tensors contain negative probabilities")

    # Normalize the distributions if they don't sum to 1
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    q = q / (q.sum(dim=1, keepdim=True) + eps)

    # Clamp values to avoid numerical issues
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)

    # Compute the mixture distribution
    m = 0.5 * (p + q)

    # Compute KL divergence for both distributions with respect to the mixture
    kl_p_m = torch.sum(
        p * (torch.log2(p + eps) - torch.log2(m + eps)), dim=1, keepdim=True
    )
    kl_q_m = torch.sum(
        q * (torch.log2(q + eps) - torch.log2(m + eps)), dim=1, keepdim=True
    )

    # Compute JS divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    # Ensure non-negative values due to numerical precision
    js_div = torch.clamp(js_div, min=0.0)

    # Return JS distance (square root of JS divergence)
    return torch.sqrt(js_div)


def diff_distributions(
    vocab_dist_m1_outputs: torch.Tensor | None = None,
    vocab_dist_m2_outputs: torch.Tensor | None = None,
    m1_outputs_path: str | None = None,
    m2_outputs_path: str | None = None,
    k: int = 10,
):
    if vocab_dist_m1_outputs is None:
        vocab_dist_m1_outputs = torch.load(
            m1_outputs_path, map_location=torch.device("cuda")
        )

    if vocab_dist_m2_outputs is None:
        vocab_dist_m2_outputs = torch.load(
            m2_outputs_path, map_location=torch.device("cuda")
        )

    js_dists = compute_js_distance(
        vocab_dist_m1_outputs, vocab_dist_m2_outputs
    ).reshape(-1)
    topk_overlaps = compute_topk_overlap(
        vocab_dist_m1_outputs, vocab_dist_m2_outputs, k
    ).reshape(-1)

    return js_dists, topk_overlaps


def print_diffs(
    js_dists, topk_overlaps, k, print_str, js_dist=False, topk_overlap=False
):
    nl = "\n"
    tab = "\t"
    qcuts = np.linspace(0, 100, 11)
    quantiles = [(i, round(js_dists.quantile(i / 100).item(), 4)) for i in qcuts]

    if js_dist:
        print(f"{print_str} JS Distance (avg = {round(js_dists.mean().item(), 4)})")
        print(
            f"{print_str} JS Distance (percentiles):{nl+tab}{(nl+tab).join([str(q) for q in quantiles])}"
        )
        js_dists_90th_mean = (
            js_dists[js_dists >= js_dists.quantile(0.9).item()].mean().item()
        )
        js_dists_90th_std = (
            js_dists[js_dists >= js_dists.quantile(0.9).item()].std().item()
        )
        print(
            f"Overall avg JS dist: {round(js_dists.mean().item(), 4)} ({round(js_dists.std().item(), 4)})"
        )
        print(
            f"Average JS dist of 90% percentile: {round(js_dists_90th_mean, 4)} ({round(js_dists_90th_std, 4)})"
        )

    if topk_overlap:
        quantiles = [
            (i, round(topk_overlaps.float().quantile(i / 100).item(), 4)) for i in qcuts
        ]
        print(
            f"{print_str} Top-{k} Overlap (avg = {round(topk_overlaps.float().mean().item(), 4)})"
        )
        print(
            f"{print_str} Top-{k} Overlap (percentiles):{nl+tab}{(nl+tab).join([str(q) for q in quantiles])}"
        )
    print("\n\n\n")


def get_indices_above_n_ptile(distances, n):
    # Get examples with distances > top-nth percentile

    qcuts = np.linspace(0, 100, n + 1)
    quantiles = [(i, round(distances.quantile(i / 100).item(), 4)) for i in qcuts]
    nth_ptile = quantiles[-2][1]
    return torch.where(distances > nth_ptile)


# import seaborn as sns


def get_topk_tokens(df, k):
    df_copy = df.copy()
    # df.set_index("token", inplace=True)
    h = df_copy.head(k).copy()
    t = df_copy.tail(k).copy()
    h.loc[:, "type"] = "full"
    t.loc[:, "type"] = "lora"
    h_med = h.freq.median()
    t_med = t.freq.median()
    # reverse the order of the tail
    t = t.iloc[::-1]
    return pd.concat([h, t], axis=0), h_med, t_med


def plt_topk_tokens(sorted_vocab_freqs, k):
    topk_tokens, h_med, t_med = get_topk_tokens(sorted_vocab_freqs, k)
    label_map = {"full": "Full", "lora": "LoRA"}
    # set label for full and lora in legend

    s = sns.scatterplot(x="token", y="freq", hue="type", data=topk_tokens, s=15)
    # Annotate Points
    for i, row in topk_tokens.iterrows():
        s.annotate(i, xy=(i, row["freq"]), xytext=(5, 5), textcoords="offset points")

    handles, labels = s.get_legend_handles_labels()
    legend_color_map = {}
    for i, label in enumerate(labels):  # Exclude the legend title
        legend_color_map[label] = handles[i].get_color()

    s.legend(handles, [label_map[label] for label in labels])
    # Add horizontal line for median frequencies
    full_median_label = f"Full (median freq={h_med})"
    lora_median_label = f"LoRA (median freq={t_med})"

    plt.axhline(
        y=h_med, linestyle="--", label=full_median_label, color=legend_color_map["full"]
    )
    plt.axhline(
        y=t_med, linestyle="--", label=lora_median_label, color=legend_color_map["lora"]
    )
    # Remove x-axis ticks
    # plt.xticks([])
    plt.yscale("log", base=2)
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()


def get_vocab_ordered_by_diff(f_output, l_output):
    diffs = f_output - l_output
    print(diffs.shape)
    vocab_diffs = diffs.mean(dim=0)
    vocab_sorted = torch.argsort(vocab_diffs, descending=True)
    return vocab_sorted


def map_ordered_vocab_to_freqs(tkzr, vocab_sorted, freq_df):
    vocab_sorted_cpu = vocab_sorted.cpu().numpy()
    vocab_sorted_tokens = tkzr.convert_ids_to_tokens(vocab_sorted_cpu)
    sorted_vocab = list(zip(vocab_sorted_tokens, vocab_sorted_cpu))
    freq_dict = {k: v["freq"] for k, v in freq_df.to_dict(orient="index").items()}
    sorted_vocab_with_freqs = [
        (token, freq_dict.get(token_id, 0)) for token, token_id in sorted_vocab
    ]
    return sorted_vocab_with_freqs


# Spearman's rank correlation coefficient


def get_corr(df, tail_length=20):
    df["full_pref_rank"] = range(1, len(df) + 1)
    topk = df[:tail_length].copy()
    bottomk = df[len(df) - tail_length :].copy()
    combined = pd.concat([topk, bottomk], axis=0)

    return spearmanr(combined["full_pref_rank"], combined["freq"])


def print_token_probs_at_index(full, lora, js_dists, dist_rank):
    i = js_dists.argsort(dim=0, descending=False)[dist_rank]
    print("rank:", dist_rank, "index:", i.item())
    order_full = full[i].argsort(dim=0, descending=True)
    ord_probs_full = full[i, order_full].cpu().numpy().round(4)

    order_lora = lora[i].argsort(dim=0, descending=True)
    ord_probs_lora = lora[i, order_lora].cpu().numpy().round(4)

    print(ord_probs_full[:5])
    print(ord_probs_lora[:5])


def print_example_at_index(tkzr, lora_output_data, full_output_data, freq_df, i, k=5):
    tkn_id = lora_output_data.loc[i].top_k_pred_tokens[0]
    prompt = lora_output_data.loc[i].in_tokens
    tkn = tkzr.convert_ids_to_tokens(tkn_id)
    lora_probs = [round(x, 3) for x in lora_output_data.loc[i].top_k_pred_probs[:k]]
    lora_tkn_ids = lora_output_data.loc[i].top_k_pred_tokens[:k]
    lora_tkns = tkzr.convert_ids_to_tokens(lora_tkn_ids)
    full_tkn_ids = full_output_data.loc[i].top_k_pred_tokens[:k]
    full_probs = [round(x, 3) for x in full_output_data.loc[i].top_k_pred_probs[:k]]
    full_tkns = tkzr.convert_ids_to_tokens(
        full_output_data.loc[i].top_k_pred_tokens[:k]
    )
    tkn_id = tkzr.convert_tokens_to_ids(tkn)
    # print(freq_df.loc[tkn])

    print(f"prompt: {prompt}")
    print(f"token: {tkn}({tkn_id})")

    print(f"lora output: {lora_probs[0]}")
    print(f"full output: {full_probs[0]}")

    nl = "\n"
    lora_probs_str = "\t" + nl.join(
        [f"<<{tkn}>>: {prob}" for tkn, prob in zip(lora_tkns, lora_probs)]
    )
    full_probs_str = "\t" + nl.join(
        [f"<<{tkn}>>: {prob}" for tkn, prob in zip(full_tkns, full_probs)]
    )

    lora_tkn_count = freq_df.loc[lora_tkn_ids].freq
    full_tkn_count = freq_df.loc[full_tkn_ids].freq

    print(len(lora_tkn_count), len(lora_probs))
    full_probs_str = nl.join(
        [
            f"{tkn} {prob} {count}"
            for tkn, prob, count in zip(full_tkns, full_probs, full_tkn_count)
        ]
    )
    lora_probs_str = nl.join(
        [
            f"{tkn} {prob} {count}"
            for tkn, prob, count in zip(lora_tkns, lora_probs, lora_tkn_count)
        ]
    )

    print(f"lora probs: \n{lora_probs_str}")
    print(f"\nfull probs: \n{full_probs_str}")


# entropy of both full and lora increases with JS distance and
#
