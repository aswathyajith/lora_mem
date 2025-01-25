# UTILITIY FUNCTIONS FOR ANALYSIS
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import random
import numpy as np
import pandas as pd
import shutil
import cv2
import os
from scipy.spatial import distance
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection

# Ordering tokens by curr_token_freq
def order_by_freq(df, var, sort_on="base_prob", ascending=False):
    ordered_df = df[["curr_token", "base_prob", var]]
    ordered_df = ordered_df.sort_values(by=sort_on, ascending=ascending)
    ordered_df = ordered_df.drop_duplicates(subset="curr_token")
    ordered_df.reset_index(drop=True, inplace=True)
    return ordered_df[["curr_token", var]]

def scatterplot(df, x, y, xlabel=r"$PMI(w_i|w_{1:i-1})$", ylabel=r"$p_{full} - p_{lora}$", legend_title="Token probability of base model", save_path=None, hue="base_prob"):
    ax = sns.scatterplot(df, x=x, y=y, hue=hue, s=5, linewidth=0)
    wrapped_title = textwrap.fill(legend_title, width=18)
    ax.legend(title=wrapped_title, bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_ft_div_all_feats(df, save_path): 
    f, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    sns.scatterplot(df, x="lora_base_diff", y="full_base_diff", hue="base_prob", s=5, linewidth=0, ax=axes[0])
    sns.scatterplot(df, x="lora_base_diff", y="full_base_diff", hue="pmi", s=5, linewidth=0, ax=axes[1])
    sns.scatterplot(df, x="lora_base_diff", y="full_base_diff", hue="curr_token_freq", s=5, linewidth=0, ax=axes[2])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def base_divergence(df, hue, diag, legend_title, plt_title=None, ax=None, save_path=None):
    scatter = sns.scatterplot(df, x="full_base_diff", y="lora_base_diff", hue=hue, s=5, linewidth=0, ax=ax)
    showplot = False
    if ax is None:
        ax = scatter
        showplot = True
    title=legend_title
    wrapped_title = textwrap.fill(title, width=18)
    ax.legend(title=wrapped_title, bbox_to_anchor=(1, 1))
    
    #Plot line along diagonal where p_full == p_lora
    if diag:
        ax.plot([-1, 1], [-1, 1], linestyle="dashed", color='r', linewidth=0.8)
    ax.set_xlabel(r"$p_{full} - p_{base}$")
    ax.set_ylabel(r'$p_{lora} - p_{base}$')
    ax.set_title(plt_title)
    if plt_title is not None: 
        ax.set_title(plt_title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if showplot:
        plt.show()

def get_region(df, a, b): 
    x1, x2, y1, y2 = a[0], a[1], b[0], b[1]
    indices = (df["full_base_diff"] > x1) & (df["full_base_diff"] < x2) & (df["lora_base_diff"] < y2) & (df["lora_base_diff"] > y1)
    return df[indices]


# Sample n points from both datasets to get less crowded plots
def sample_df(df, n=50000):
    return df.loc[random.sample(range(len(df)), n)]

def plot_fig1(save_path="results/figs/fig1.pdf"):
    plt.plot([-1, 1], [-1, 1], linestyle="dashed", color='r', linewidth=0.8)
    plt.plot([-1, 0], [0, 1], linestyle="dashed", color='gray', linewidth=0.8)
    plt.plot([0, 1], [-1, 0], linestyle="dashed", color='gray', linewidth=0.8)

    plt.annotate(r'$p_{full} - p_{lora} = 0$', xy=(0, -0.1),  
                ha='center', va='bottom', rotation=35)
    plt.annotate(r'$p_{full} - p_{lora} = -1$', xy=(-0.57, 0.3),  
                ha='center', va='bottom', rotation=35)
    plt.annotate(r'$p_{full} - p_{lora} = 1$', xy=(0.4, -0.86),  
                ha='center', va='bottom', rotation=35)


    # Add colored strip along x-axis
    strip_height = 0.25  # Height of the colored strip

    colors = sns.color_palette("pastel")

    x = np.array([-0.5, 0, 0.5])
    y_bottom = np.array([0.75-strip_height, 0.75-strip_height, 0.75-strip_height])
    y_top = np.array([0.75-strip_height, 0.75+strip_height, 0.75+strip_height])
    plt.fill_between(x, y_bottom, y_top, color=colors[0], alpha=0.3)
    wrapped_text = '\n'.join([r'$\it{' + '\ '.join(line.split()) + '}$' for line in textwrap.wrap('LoRA regurgitates', width=10, break_long_words=False)])
    plt.annotate(wrapped_text, xy=(0.25, 0.75), ha='center', va='center', fontsize=8)

    x = np.array([0.5, 1])
    y_bottom = np.array([0.75-strip_height, 0.75-strip_height])
    y_top = np.array([0.75+strip_height, 0.75+strip_height])
    plt.fill_between(x, y_bottom, y_top, color=colors[1], alpha=0.3)
    wrapped_text = '\n'.join([r'$\it{' + '\ '.join(line.split()) + '}$' for line in textwrap.wrap('Finetuning regurgitates', width=10, break_long_words=False)])
    plt.annotate(wrapped_text, xy=(0.75, 0.75), ha='center', va='center', fontsize=8)

    x = np.array([0.5, 0.5, 1])
    y_bottom = np.array([strip_height-0.75, strip_height-0.75, 0.25 - strip_height])
    y_top = np.array([0.5, 0.5, 0.5])
    plt.fill_between(x, y_bottom, y_top, color=colors[2], alpha=0.3)
    wrapped_text = '\n'.join([r'$\it{' + '\ '.join(line.split()) + '}$' for line in textwrap.wrap('Full regurgitates', width=10, break_long_words=False)])
    plt.annotate(wrapped_text, xy=(0.75, 0.2), ha='center', va='center', fontsize=8)

    # Base model regurgitates
    x = np.array([-1, -0.5, -0.5, 0, 0.5])
    y_top = np.array([0, 0.5, -0.5, -0.5, -0.5])
    y_bottom = np.array([-1, -1, -1, -1, -0.5])
    plt.fill_between(x, y_bottom, y_top, color=colors[3], alpha=0.3)
    wrapped_text = '\n'.join([r'$\it{' + '\ '.join(line.split()) + '}$' for line in textwrap.wrap('Base model infers', width=10, break_long_words=False)])
    plt.annotate(wrapped_text, xy=(-0.75, -0.6), ha='center', va='center', fontsize=8)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel(r"$p_{full} - p_{base}$")
    plt.ylabel(r'$p_{lora} - p_{base}$')
    plt.savefig("results/figs/fig1.pdf", bbox_inches='tight')
    plt.show()

# Function to bin a variable in the dataframe
def bin_var(df, var, abs=True, n=4):
    var_vals = df[var]
    if abs:
        var_vals = np.abs(var_vals)

    percentiles_at = np.linspace(0, 1, n+1)

    bins = list(var_vals.quantile(percentiles_at))
    labels = [f"Q{i+1}: {round(bins[i], 3)}-{round(bins[i+1], 3)}" for i in range(len(bins)-1)]
    df[f"{var}_bins"] = pd.cut(var_vals, bins=bins, labels=labels, include_lowest=True)
    return df

def create_jointplot(df, title, title_fontsize=10, x="full_base_diff", y="lora_base_diff", hue="token_in_context"):

   # First subplot: JointGrid for 'total_bill' and 'tip'
   g1 = sns.JointGrid(data=df, x=x, y=y, hue=hue, height=4, dropna=True)
   g1.plot_joint(sns.scatterplot,s=4,linewidth=0)

   # g1.ax_joint.set_ylim(-0.2,1)
   g1.ax_marg_x.set_xlim(0,1)
   g1.figure.suptitle(title, fontsize=title_fontsize, y=0.95)
   g1.ax_marg_y.set_ylim(0,1)

   g1.plot_marginals(sns.kdeplot, multiple="stack", fill=True, common_norm=False, linewidth=0)
   handles, labels = g1.ax_joint.get_legend_handles_labels()
   if g1.ax_joint.legend_ is not None:
      g1.ax_joint.legend_.remove()

   plt.tight_layout()
   
   return plt, handles, labels

def create_split_jointplot(data, figsize=(12, 12), title=None, title_fontsize=10, kde=True, save_path=None):
    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    
    # Create a more complex gridspec
    gs = fig.add_gridspec(5, 5)
    
    # Create the scatter plot in the center
    ax_scatter = fig.add_subplot(gs[1:4, 1:4])

    # Define colors for token_in_context
    colors = {True: 'darkorange', False: 'dodgerblue'}
    
    # Set plot limits
    xlim = (0, 1)
    ylim = (0, 1)

    # Create marginal plots for points above diagonal
    ax_histx_above = fig.add_subplot(gs[0, 1:4])  # top marginal
    ax_histy_above = fig.add_subplot(gs[1:4, 0])  # left marginal
    
    # Create marginal plots for points below diagonal
    ax_histx_below = fig.add_subplot(gs[4, 1:4])  # bottom marginal
    ax_histy_below = fig.add_subplot(gs[1:4, 4])  # right marginal
    
    # Split data into above and below diagonal
    above_diagonal = data[data['lora_base_diff'] > data['full_base_diff']]
    below_diagonal = data[data['lora_base_diff'] <= data['full_base_diff']]
    kde_above_xlim = (min(above_diagonal['full_base_diff']), max(above_diagonal['full_base_diff']))
    kde_below_xlim = (min(below_diagonal['full_base_diff']), max(below_diagonal['full_base_diff']))
    kde_above_ylim = (min(above_diagonal['lora_base_diff']), max(above_diagonal['lora_base_diff']))
    kde_below_ylim = (min(below_diagonal['lora_base_diff']), max(below_diagonal['lora_base_diff']))
    
    # Scatter plot with token_in_context coloring
    for is_in_context in [True, False]:
        # Points below diagonal
        mask_below = below_diagonal['token_in_context'] == is_in_context
        below_context = below_diagonal[mask_below]
        ax_scatter.scatter(below_diagonal[mask_below]['full_base_diff'], 
                         below_diagonal[mask_below]['lora_base_diff'],
                        #  alpha=0.5, 
                         color=colors[is_in_context],
                         linewidth=0,
                         s=10,
                         label=f'Token{"" if is_in_context else " not"} in context')
        
        # Points above diagonal
        mask_above = above_diagonal['token_in_context'] == is_in_context
        above_context = above_diagonal[mask_above]
        ax_scatter.scatter(above_diagonal[mask_above]['full_base_diff'],
                         above_diagonal[mask_above]['lora_base_diff'],
                        #  alpha=0.5,
                         s=10,
                         color=colors[is_in_context], 
                         linewidth=0)
    
        if kde: 
            sns.kdeplot(data=above_context['lora_base_diff'], ax=ax_histy_above,
                    color=colors[is_in_context], alpha=0.5, fill=True,
                    vertical=True, clip=kde_above_ylim)
                    # KDE plots for points above diagonal (top and right)
            sns.kdeplot(data=above_context['full_base_diff'], ax=ax_histx_above,
                   color=colors[is_in_context], alpha=0.5, fill=True,
                   label=f'{"In" if is_in_context else "Not in"} context',
                    clip=kde_above_xlim)
            
            # KDE plots for points below diagonal (bottom and left)
            sns.kdeplot(data=below_context['full_base_diff'], ax=ax_histx_below,
                    color=colors[is_in_context], alpha=0.5, fill=True,
                    clip=kde_below_xlim)
            sns.kdeplot(data=below_context['lora_base_diff'], ax=ax_histy_below,
                    color=colors[is_in_context], alpha=0.5, fill=True,
                    vertical=True, clip=kde_below_ylim)
        else: 
            
            ax_histx_above.hist(above_context['full_base_diff'], bins=50, alpha=0.5, color=colors[is_in_context])
            ax_histy_above.hist(above_context['lora_base_diff'], bins=50, orientation='horizontal', alpha=0.5, color=colors[is_in_context])
            
            # Marginal distributions for points below diagonal (bottom and left)
            ax_histx_below.hist(below_context['full_base_diff'], bins=50, alpha=0.5, color=colors[is_in_context])
            ax_histy_below.hist(below_context['lora_base_diff'], bins=50, orientation='horizontal', alpha=0.5, color=colors[is_in_context])
    
    # Draw diagonal line
    ax_scatter.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # Set limits for all plots
    ax_scatter.set_xlim(xlim)
    ax_scatter.set_ylim(ylim)

    # Invert the left and bottom marginal
    ax_histy_above.invert_xaxis()
    ax_histx_below.invert_yaxis()

    # Labels and title
    ax_scatter.set_xlabel('full_base_diff')
    ax_scatter.set_ylabel('lora_base_diff')
    
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize, y=0.95)
    
    # Remove all spines, ticks, and labels from marginal plots
    for ax in [ax_histx_above, ax_histx_below, ax_histy_above, ax_histy_below]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add legend to top marginal
    handles, labels = ax_histx_above.get_legend_handles_labels()
    
    # Align limits
    ax_histx_above.set_xlim(ax_scatter.get_xlim())
    ax_histx_below.set_xlim(ax_scatter.get_xlim())
    ax_histy_above.set_ylim(ax_scatter.get_ylim())
    ax_histy_below.set_ylim(ax_scatter.get_ylim())
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    return fig, handles, labels

def stitch_plots(dfs, titles, figsize=(90,30), save_path=None):
   # stitch plots together
   fig, axarr = plt.subplots(1,len(dfs), figsize=figsize)
   tmp_dir = "results/plots/finetuning/full/tmp"
   os.makedirs(tmp_dir, exist_ok=True)
   for i, df in enumerate(dfs):
      
      g, handles, labels = create_jointplot(df, title=titles[i])
      filepath=os.path.join(tmp_dir, f"image_{i}.png")
      g.savefig(filepath)
      g.close()

      im = cv2.imread(filepath)
      axarr[i].imshow(im)
      axarr[i].axis("off")


   fig.legend(handles, ["Token not in context", "Token in context"], loc='lower center', bbox_to_anchor=(0.5, -.1), fontsize=100, ncol=2, markerscale=20)
   plt.subplots_adjust(wspace=0, hspace=0, top=0.6)
   plt.tight_layout()
   if save_path is not None:
      plt.savefig(save_path, bbox_inches='tight')
   shutil.rmtree(tmp_dir)
   plt.show()

def stitch_plots_tmp(filepaths, handles, labels, figsize=(90,30), save_path=None):
   # stitch plots together
   fig, axarr = plt.subplots(1,len(filepaths), figsize=figsize)
   tmp_dir = "results/plots/finetuning/full/tmp"
   os.makedirs(tmp_dir, exist_ok=True)
   for i, filepath in enumerate(filepaths):

      im = cv2.imread(filepath)
      axarr[i].imshow(im)
      axarr[i].axis("off")

   new_handles = []
   # Update PolyCollection legend 
   for handle in handles:
       print(handle)
       new_handle = PolyCollection([])
       new_handle.update_from(handle)
       new_handles.append(new_handle)
       
   fig.legend(new_handles, labels, fontsize=30, ncol=2, markerscale=20)
   plt.subplots_adjust(wspace=0, hspace=0, top=0.6)
   print(new_handles, labels)
   plt.tight_layout()
   if save_path is not None:
      plt.savefig(save_path, bbox_inches='tight')
#    shutil.rmtree(tmp_dir)
   plt.show()


def plot_ntile_vocab_dist(df, top_k=100, sort_on="base_prob", ascending=False, num_bins=4):

    # create a plot with num_rows x num_cols subplots
    num_rows = num_bins + 1
    num_cols = 3

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))

    # Flatten axs for easier indexing
    axs = axs.flatten()

    # create curr_token_freq_bins
    q_i_bins = bin_var(df=df, var=sort_on, n=num_bins)

    # Get unique values of curr_token_freq_bins
    q_i_bins = q_i_bins.sort_values(by=sort_on)[f'{sort_on}_bins'].unique()

    curr_token_freq_sorted_all = order_by_freq(df, var="curr_token_freq", sort_on="curr_token_freq", ascending=ascending)
    pt_curr_token_freq_sorted_all = order_by_freq(df, var="pt_curr_token_freq", sort_on="pt_curr_token_freq", ascending=ascending)
    
    # Limit to top 50 tokens for clarity
    curr_token_freq_sorted = curr_token_freq_sorted_all.loc[:top_k]
    pt_curr_token_freq_sorted = pt_curr_token_freq_sorted_all.loc[:top_k]

    all_sorted_dfs = [curr_token_freq_sorted_all, pt_curr_token_freq_sorted_all]
    sorted_dfs = [curr_token_freq_sorted, pt_curr_token_freq_sorted]#, rel_prev_sorted]
    vars = ["curr_token_freq", "pt_curr_token_freq"]#, "rel_prev"]

    # set titles
    axs[1].set_title("FT corpus token frequency")
    axs[2].set_title("PT corpus token frequency")

    sns.ecdfplot(data=curr_token_freq_sorted, x="curr_token_freq", ax=axs[1])
    sns.ecdfplot(data=pt_curr_token_freq_sorted, x="pt_curr_token_freq", ax=axs[2])

    for idx in [1, 2]:
        axs[idx].set_xlabel('')
        axs[idx].set_xticks([])

    for j in range(len(sorted_dfs)):
        for i in range(num_rows):
            # get index in flattened array for first column
            idx = 1 + i * num_cols + j
            
            if (idx - 1) % num_cols == 0: 
                # no axes
                axs[idx-1].set_axis_off()
                if i != 0:
                    axs[idx-1].text(0.5, 0.5, q_i_bins[i-1], ha='center', va='center', fontsize=20)
                else:
                    axs[idx-1].text(0.5, 0.5, "Across all bins", ha='center', va='center', fontsize=20)
            # skip first row
            if i == 0: 
                continue

            q_i_df = df[df[f'{sort_on}_bins'] == q_i_bins[i-1]][["curr_token"]]
            
            # add count of curr_token in quant_df
            q_i_token_counts = q_i_df["curr_token"].value_counts().reset_index()
            q_i_token_counts.columns = ["curr_token", vars[j]]

            # sort the quant_df_sorted by order of curr_token curr_token_freq_sorted
            quant_df_sorted = sorted_dfs[j][["curr_token"]].merge(q_i_token_counts, on="curr_token", how="left")
            
            
            # replace nan with 0
            quant_df_sorted.fillna(0, inplace=True)
            sns.ecdfplot(data=quant_df_sorted, x=f'{vars[j]}', ax=axs[idx])

            # remove x labels
            axs[idx].set_xlabel('')

            # Remove ticks
            axs[idx].set_xticks([])

            # measure JS divergence between curr_token_freq of curr_token_freq_sorted_all and q_i_token_counts
            
            q_i_token_counts = curr_token_freq_sorted_all[["curr_token"]].merge(q_i_token_counts, on="curr_token", how="left")
            q_i_token_counts.fillna(0, inplace=True) # replace nan with 0
            P = all_sorted_dfs[j][vars[j]] / all_sorted_dfs[j][vars[j]].sum()
            Q = q_i_token_counts[vars[j]] / q_i_token_counts[vars[j]].sum()
            js_dist = distance.jensenshannon(P, Q)

            
            # 
            axs[idx].text(0.8, 0.8, f'JS dist: {js_dist:.2f}', ha='right', va='top', fontsize=10, transform=axs[idx].transAxes)

    plt.tight_layout()
    plt.show()