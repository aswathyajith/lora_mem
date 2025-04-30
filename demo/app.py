import streamlit as st
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.colors as mcolors
import html
import argparse

def load_json_file(file_path: str) -> Dict:
    return pd.read_json(file_path, orient='records', lines=True)

def calculate_token_overlap(lora_tokens: List[str], full_tokens: List[str], k: int = 5) -> float:
    """Calculate the IoU score between top-k predictions."""

    # get the top k tokens
    lora_top_k = set(lora_tokens[:k])
    full_top_k = set(full_tokens[:k])
    
    assert len(lora_top_k) == len(full_top_k) == k, f"lora_top_k: {len(lora_top_k)}, full_top_k: {len(full_top_k)}, k: {k}"
    intersection = len(lora_top_k.intersection(full_top_k))
    union = len(lora_top_k.union(full_top_k))
    return round(intersection / union, 2)

def get_color_from_score(score: float) -> str:
    """Convert IoU score to a color in hex format."""
    # Create a color gradient from red (0) to green (1)
    color = mcolors.to_hex([1 - score, score, 0])
    return color

def main():
    st.title("Model Outputs Visualization")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_outputs", type=str, default="demo/data/math.json")
    args = parser.parse_args()

    # Load the data
    path_to_outputs = args.path_to_outputs
    
    try:
        data = load_json_file(path_to_outputs)
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return
    
    # Get the first example for visualization
    n = data.seq_id.values.max()
    example_idx = st.slider("Select example index", 0, n, 0)
    
    example = data[data.seq_id == example_idx]
    # Add custom CSS for tokens
    st.markdown("""
        <style>
        .token {
            display: inline-block;
            padding: 2px 4px;
            border-radius: 3px;
            margin: 0 1px;
            position: relative;
        }
        .tooltip {
            visibility: hidden;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 200%;
            transform: translateX(-50%);
            color: white;
            text-align: left;
            padding: 5px;
            border-radius: 3px;
            border: 1px solid #ccc;
            white-space: pre-line;
            font-family: monospace;
            width: 600px;
            overflow: visible;
        }
        .token:hover .tooltip {
            visibility: visible;
        }
        .tooltip table th {
            padding: 6px;
            border: 1px solid #ddd;
            border-bottom: 2px solid #666;
            color: black;
            width: 33%;
        }
        .tooltip table td {
            padding: 4px;
            border: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create HTML for tokens with colors and tooltips
    k = st.slider("Top-k tokens to compare", 1, 10, 5)
    tokens_html = []
    
    st.subheader("Context")
    for _, row in example.iterrows():
        iou_score = calculate_token_overlap(
            row["lora_r16_top_k_pred_tokens"], 
            row["full_top_k_pred_tokens"], 
            k
        )
        color = get_color_from_score(iou_score)
        
        # Create tooltip content

        top_k_pred_tokens_lora = row["lora_r16_top_k_pred_tokens"][:k]
        top_k_pred_probs_lora = row["lora_r16_top_k_pred_probs"][:k]
        lora_pairs = [f"{tkn} ({round(prob, 2)})" for tkn, prob in zip(top_k_pred_tokens_lora, top_k_pred_probs_lora)]
        top_k_pred_tokens_full = row["full_top_k_pred_tokens"][:k]
        top_k_pred_probs_full = row["full_top_k_pred_probs"][:k]
        full_pairs = [f"{tkn} ({round(prob, 2)})" for tkn, prob in zip(top_k_pred_tokens_full, top_k_pred_probs_full)]
        base_pairs = [f"{tkn} ({round(prob, 2)})" for tkn, prob in zip(row["base_top_k_pred_tokens"][:k], row["base_top_k_pred_probs"][:k])]

        rows = "\n".join([f"<tr><td>{base}</td><td>{full}</td><td>{lora}</td></tr>" for base, full, lora in zip(base_pairs, full_pairs, lora_pairs)])
        # Create tooltip content
        tooltip_content = """
            <table style='width:100%; border-collapse:collapse;'>
                <tr><th style='background:{color};'>Base Model</th><th style='background:{color};'>Full Model</th><th style='background:{color};'>LoRA (r=16, attn_only) Model</th></tr>
                {rows}
                <tr><td colspan='3' style='text-align:center; background:{color}; color: black;'><b>IoU Score: {iou_score}</b></td></tr>
            </table>
            """.format(
                        color=color,
                        iou_score=iou_score,
                        rows=rows
                    )
        
        # Create token HTML with color and tooltip
        token_html = f'<div class="token" style="background-color: {color}">{row["curr_token"]}<span class="tooltip" style="background-color: black">{tooltip_content}</span></div>'

        tokens_html.append(token_html)
    
    # Display tokens
    st.markdown("".join(tokens_html), unsafe_allow_html=True)
    
    # Add a color scale legend
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("IoU Score Legend")
    st.markdown("""
        <div style="display: flex; justify-content: space-between; margin-top: 20px;">
            <span>0.0</span>
            <span>0.5</span>
            <span>1.0</span>
        </div>
        <div style="height: 20px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); margin: 10px 0;"></div>
    """, unsafe_allow_html=True)
    
    # Display raw data for debugging
    if st.checkbox("Show raw data"):
        st.json({
            "example": example.to_dict()
        })

if __name__ == "__main__":
    main()