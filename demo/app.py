import tracemalloc
tracemalloc.start()
import streamlit as st
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.colors as mcolors
import html
import argparse
import matplotlib.pyplot as plt
import sys

@st.cache_data
def load_json_file(file_path: str) -> Dict:
    # normalize entropies to [0, 1]
    data = pd.read_json(file_path, orient='records', lines=True)
    data['base_entropy_color'] = 1 - (data['base_entropy'] - data['base_entropy'].min()) / (data['base_entropy'].max() - data['base_entropy'].min())
    data['lora_r16_entropy_color'] = 1 - (data['lora_r16_entropy'] - data['lora_r16_entropy'].min()) / (data['lora_r16_entropy'].max() - data['lora_r16_entropy'].min())
    data['full_entropy_color'] = 1 - (data['full_entropy'] - data['full_entropy'].min()) / (data['full_entropy'].max() - data['full_entropy'].min())
    return data

@st.cache_data
def calculate_token_overlap(token_list1: List[str], token_list2: List[str], k: int = 5) -> float:
    """Calculate the IoU score between top-k predictions."""

    # get the top k tokens
    token_list1_top_k = set(token_list1[:k])
    token_list2_top_k = set(token_list2[:k])
    
    assert len(token_list1_top_k) == len(token_list2_top_k) == k, f"token_list1_top_k: {len(token_list1_top_k)}, token_list2_top_k: {len(token_list2_top_k)}, k: {k}"
    intersection = len(token_list1_top_k.intersection(token_list2_top_k))
    union = len(token_list1_top_k.union(token_list2_top_k))
    return round(intersection / union, 2)

@st.cache_data
def get_color_from_value(value: float, cmap: str = 'RdYlGn') -> str:
    """Convert a value to a color in hex format using the specified colormap."""
    # Normalize value to [0,1] if needed
    if value > 1:
        value = value / 100
    color = mcolors.to_hex(plt.get_cmap(cmap)(value))
    return color

@st.cache_data
def get_comparison_color(full_prob: float, lora_prob: float) -> str:
    """Get color for full vs lora probability comparison."""
    if full_prob > lora_prob:
        return '#2471a3'  # Muted blue-gray
    elif lora_prob > full_prob:
        return '#dc7633'  # Muted golden brown
    else:
        return '#797d7f'  # Black

@st.cache_data
def get_custom_css() -> str:
    return """
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
            position: fixed;
            z-index: 1000;
            right: 50px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            text-align: left;
            padding: 5px;
            border-radius: 3px;
            border: 1px solid #ccc;
            white-space: pre-line;
            font-family: monospace;
            width: 500px;
            overflow: visible;
            background-color: black;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
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
    """

@st.cache_data
def get_tooltip_template() -> str:
    return """
    <table style='width:100%; border-collapse:collapse;'>
        <tr><th style='background:{color};'>Base Model</th><th style='background:{color};'>Full Model</th><th style='background:{color};'>LoRA (r=16, attn_only) Model</th></tr>
        {rows}
        <tr><td colspan='3' style='text-align:center; background:{color}; color: black;'><b>{metric}: {value:.3f}</b></td></tr>
    </table>
    """

@st.cache_data
def generate_token_html(token: str, color: str, tooltip_content: str) -> str:
    return f'<div class="token" style="background-color: {color}">{token}<span class="tooltip" style="background-color: black">{tooltip_content}</span></div>'

@st.cache_data
def get_example_data(data: pd.DataFrame, example_idx: int) -> pd.DataFrame:
    """Get and process example data with caching."""
    return data[data.seq_id == example_idx].copy()

@st.cache_data
def get_panel_names(panels):
    """Get panel names in a cached manner."""
    return [panel['name'] for panel in panels]

def update_panel_name(idx: int):
    """Update panel name without triggering a full rerun."""
    key = f"name_{idx}"
    if key in st.session_state:
        st.session_state.panels[idx]['name'] = st.session_state[key]

def delete_panel(idx: int):
    """Delete panel without triggering unnecessary reruns."""
    if len(st.session_state.panels) > 1:
        st.session_state.panels.pop(idx)
        # st.experimental_rerun()

@st.cache_data
def process_example_tokens(example: pd.DataFrame, color_by: str, color_by_options: dict, k: int) -> list:
    """Process and cache token computations for an example."""
    tokens_html = []
    tooltip_template = get_tooltip_template()
    
    for _, row in example.iterrows():
        # Get the coloring value based on selection
        if color_by_options[color_by] == 'lora_full_overlap':
            color_value = calculate_token_overlap(
                row["lora_r16_top_k_pred_tokens"], 
                row["full_top_k_pred_tokens"], 
                k
            )
            color = get_color_from_value(color_value)
        elif color_by_options[color_by] == 'lora_base_overlap':
            color_value = calculate_token_overlap(
                row["lora_r16_top_k_pred_tokens"], 
                row["base_top_k_pred_tokens"], 
                k
            )
            color = get_color_from_value(color_value)
        elif color_by_options[color_by] == 'full_base_overlap':
            color_value = calculate_token_overlap(
                row["full_top_k_pred_tokens"], 
                row["base_top_k_pred_tokens"], 
                k
            )
            color = get_color_from_value(color_value)
        elif color_by_options[color_by] in ['base_entropy', 'lora_r16_entropy', 'full_entropy']:
            color = get_color_from_value(row[f"{color_by_options[color_by]}_color"])
            color_value = row[color_by_options[color_by]]
        elif color_by_options[color_by] in ['base_prob', 'lora_r16_prob', 'full_prob']:
            value_map = {
                'base_prob': row["base_prob"],
                'lora_r16_prob': row["lora_r16_prob"],
                'full_prob': row["full_prob"]
            }
            color_value = value_map[color_by_options[color_by]]
            color = get_color_from_value(color_value)
        else:  # full_vs_lora rank comparison
            full_rank = row["full_curr_token_rank"]
            lora_rank = row["lora_r16_curr_token_rank"]
            color = get_comparison_color(full_rank, lora_rank)
            color_value = int(np.abs(full_rank - lora_rank))
        
        # Pre-process token pairs
        base_pairs = [f"{tkn} ({round(prob, 2)})" for tkn, prob in zip(row["base_top_k_pred_tokens"][:k], row["base_top_k_pred_probs"][:k])]
        full_pairs = [f"{tkn} ({round(prob, 2)})" for tkn, prob in zip(row["full_top_k_pred_tokens"][:k], row["full_top_k_pred_probs"][:k])]
        lora_pairs = [f"{tkn} ({round(prob, 2)})" for tkn, prob in zip(row["lora_r16_top_k_pred_tokens"][:k], row["lora_r16_top_k_pred_probs"][:k])]
        
        rows = "\n".join([f"<tr><td>{base}</td><td>{full}</td><td>{lora}</td></tr>" for base, full, lora in zip(base_pairs, full_pairs, lora_pairs)])
        
        tooltip_content = tooltip_template.format(
            color=color,
            value=color_value,
            metric=color_by,
            rows=rows
        )
        
        tokens_html.append(generate_token_html(row["curr_token"], color, tooltip_content))
    
    return tokens_html

def get_legend_html(color_by: str) -> str:
    """Get the appropriate legend HTML based on the visualization type."""
    if color_by == 'Token Rank Comparison':
        return """
            <div style="display: flex; align-items: center; gap: 20px; margin-top: 20px;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 20px; height: 20px; background-color: #2471a3;"></div>
                    <span>LoRA ranks higher than full</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 20px; height: 20px; background-color: #dc7633;"></div>
                    <span>Full ranks higher than LoRA</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 20px; height: 20px; background-color: #797d7f; border: 1px solid #000;"></div>
                    <span>Equal</span>
                </div>
            </div>
        """
    elif 'Entropy' in color_by:
        return """
            <div style="margin-top: 20px;">
                <div style="display: flex; flex-direction: column; align-items: center; gap: 5px;">
                    <div style="width: 300px; height: 20px; background: linear-gradient(to right, #1a9850, #fff3b0, #d73027); border-radius: 4px;"></div>
                    <div style="display: flex; justify-content: space-between; width: 300px;">
                        <span>Low Entropy<br>(More Certainty)</span>
                        <span>High Entropy<br>(More Uncertainty)</span>
                    </div>
                </div>
            </div>
        """
    elif 'Probability' in color_by:
        return """
            <div style="margin-top: 20px;">
                <div style="display: flex; flex-direction: column; align-items: center; gap: 5px;">
                    <div style="width: 300px; height: 20px; background: linear-gradient(to right, #d73027, #fff3b0, #1a9850); border-radius: 4px;"></div>
                    <div style="display: flex; justify-content: space-between; width: 300px;">
                        <span>Low<br>Probability</span>
                        <span>High<br>Probability</span>
                    </div>
                </div>
            </div>
        """
    elif 'Overlap' in color_by:
        return """
            <div style="margin-top: 20px;">
                <div style="display: flex; flex-direction: column; align-items: center; gap: 5px;">
                    <div style="width: 300px; height: 20px; background: linear-gradient(to right, #d73027, #fff3b0, #1a9850); border-radius: 4px;"></div>
                    <div style="display: flex; justify-content: space-between; width: 300px;">
                        <span>Low<br>Overlap</span>
                        <span>High<br>Overlap</span>
                    </div>
                </div>
            </div>
        """
    return ""

def main():
    st.title("Model Outputs Visualization")

    # Add custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        try:
            path_to_outputs = "demo/data/math.json"
            if len(sys.argv) > 1:
                parser = argparse.ArgumentParser()
                parser.add_argument("--path_to_outputs", type=str, default="demo/data/math.json")
                args = parser.parse_args()
                path_to_outputs = args.path_to_outputs
            
            st.session_state.data = load_json_file(path_to_outputs)
        except Exception as e:
            st.error(f"Error loading data files: {e}")
            return

    if 'panels' not in st.session_state:
        st.session_state.panels = [{
            'color_by': 'Base Model Probability',
            'name': f"Panel 1"
        }]
    
    # Add panel button
    if st.button("Add New Panel"):
        panel_num = len(st.session_state.panels) + 1
        st.session_state.panels.append({
            'color_by': 'Base Model Probability',
            'name': f"Panel {panel_num}"
        })
    
    # Get the first example for visualization
    n = st.session_state.data.seq_id.values.max()
    example_idx = st.slider("Select example index", 0, n, 0)
    
    # Get cached example data
    example = get_example_data(st.session_state.data, example_idx)
    
    # Add color-by selector
    color_by_options = {
        'LoRA vs Full Overlap': 'lora_full_overlap',
        'LoRA vs Base Overlap': 'lora_base_overlap',
        'Full vs Base Overlap': 'full_base_overlap',
        'Base Model Probability': 'base_prob',
        'LoRA Model Probability': 'lora_r16_prob',
        'Full Model Probability': 'full_prob',
        'Token Rank Comparison': 'full_vs_lora',
        'Base Model Entropy': 'base_entropy',
        'LoRA Model Entropy': 'lora_r16_entropy',
        'Full Model Entropy': 'full_entropy'
    }
    
    # Initialize session state for name changes if not exists
    if 'name_changes' not in st.session_state:
        st.session_state.name_changes = {}
    
    # Slider for top-k (shared across all panels)
    k = st.slider("Top-k tokens to compare", 1, 10, 5)
    
    # Create tabs for each panel
    tabs = st.tabs([panel['name'] for panel in st.session_state.panels])
    
    # Process each panel in its tab
    for idx, (tab, panel) in enumerate(zip(tabs, st.session_state.panels)):
        with tab:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Color-by selector for this panel
                color_by = st.selectbox(
                    "Color tokens by:",
                    options=list(color_by_options.keys()),
                    key=f"color_by_{idx}",
                    index=list(color_by_options.keys()).index(panel['color_by'])
                )
                # Update panel state
                panel['color_by'] = color_by
            
            with col2:
                # Rename panel input with callback
                st.text_input(
                    "Panel name",
                    value=panel['name'],
                    key=f"name_{idx}",
                    on_change=update_panel_name,
                    args=(idx,)
                )
            
            with col3:
                # Delete panel button (prevent deleting last panel)
                if len(st.session_state.panels) > 1:
                    st.button("Delete Panel", key=f"delete_{idx}", on_click=delete_panel, args=(idx,))
            
            # Process and display tokens with caching
            tokens_html = process_example_tokens(example, color_by, color_by_options, k)
            st.markdown("".join(tokens_html), unsafe_allow_html=True)
            
            # Display legend
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"{panel['name']}: {color_by} Legend")
            st.markdown(get_legend_html(color_by), unsafe_allow_html=True)

    # Display raw data for debugging
    if st.checkbox("Show raw data"):
        st.json({
            "example": example.to_dict()
        })

if __name__ == "__main__":
    main()