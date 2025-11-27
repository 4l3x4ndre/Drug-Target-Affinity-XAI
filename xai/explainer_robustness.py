import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import streamlit as st
from tqdm import tqdm
import plotly.graph_objects as go
from MGraphDTA.regression.preprocessing import VOCAB_PROTEIN

from src.utils.plot_utils import set_plot_style, get_styled_figure_ax, style_legend, adjust_plot_limits
DATASET_COLORS = ['#95BB63', '#7E7EA5', '#487C7D', '#EA805D', '#4897DC', '#6a408d', '#c04a2e']

def test_robustness_masking(
    explainer,
    smiles_input: str,
    protein_input: str,
    per_layer_cams: np.ndarray,
    original_pred: float,
    top_k_candidates: int = 200,
    top_k_masked: int = 120,
    mask_tokens: list = None,
    n_repeats: int = 10,
    random_seed: int = 42,
):
    """
    Assess model robustness by randomly masking a subset of top importance candidates
    versus fully random residues. Suitable for deterministic models.

    Parameters
    ----------
    explainer : object
        Model explainer providing `.explain(smiles, protein, idx)` → dict with 'prediction'
    smiles_input : str
        Input drug SMILES string
    protein_input : str
        Protein sequence
    per_layer_cams : np.ndarray
        Layer-wise CAM or importance maps, shape (n_layers, seq_len)
    original_pred : float
        Original model prediction before masking
    top_k_candidates : int
        Number of top amino acids to select as candidates for perturbation
    top_k_masked : int
        Number of residues to mask among the candidates
    mask_tokens : list[str]
        List of tokens to use for masking
    n_repeats : int
        Number of random selections to perform
    random_seed : int
        Random seed for reproducibility
    """
    explainer.set_mode('train') # MC dropout

    if mask_tokens is None:
        mask_tokens = ['-'] + list(VOCAB_PROTEIN.keys()) # ['-', 'A', 'G', 'X']
        print(f"Using default mask tokens: {mask_tokens}")

    rng = np.random.default_rng(random_seed)
    seq_len = len(protein_input)

    # Aggregate importance scores across layers
    meaned_cam = np.mean(per_layer_cams, axis=0)
    cam_1d = meaned_cam.flatten()

    # Select top candidates
    top_candidate_indices = np.argsort(cam_1d)[-top_k_candidates:][::-1]

    results = []

    for mask_token in mask_tokens:
        top_masked_preds = []
        random_masked_preds = []

        for rep in tqdm(range(n_repeats), desc=f"Masking with token '{mask_token}'"):
            # --- Mask subset of top candidates ---
            masked_seq_top = list(protein_input)
            masked_indices_top = rng.choice(top_candidate_indices, size=top_k_masked, replace=False)
            for idx in masked_indices_top:
                masked_seq_top[idx] = mask_token
            masked_seq_str_top = ''.join(masked_seq_top)
            pred_top = explainer.explain(smiles_input, masked_seq_str_top, st.session_state.sample_idx)['prediction']
            top_masked_preds.append(pred_top)

            # --- Mask fully random residues ---
            masked_seq_rand = list(protein_input)
            rand_indices = rng.choice(seq_len, size=top_k_masked, replace=False)
            for idx in rand_indices:
                masked_seq_rand[idx] = mask_token
            masked_seq_str_rand = ''.join(masked_seq_rand)
            pred_rand = explainer.explain(smiles_input, masked_seq_str_rand, st.session_state.sample_idx)['prediction']
            random_masked_preds.append(pred_rand)

        # Convert to arrays
        top_masked_preds = np.array(top_masked_preds)
        random_masked_preds = np.array(random_masked_preds)

        # Compute deltas
        delta_top = top_masked_preds - original_pred
        delta_rand = random_masked_preds - original_pred

        # Paired t-test
        t_stat, p_val = ttest_rel(delta_top, delta_rand)

        results.append({
            "Mask Token": mask_token,
            "Mean Prediction (Top Subset)": top_masked_preds.mean(),
            "Std Prediction (Top Subset)": top_masked_preds.std(),
            "Mean Δ Prediction (Top Subset)": delta_top.mean(),
            "Std Δ Prediction (Top Subset)": delta_top.std(),
            "Mean Prediction (Random)": random_masked_preds.mean(),
            "Std Prediction (Random)": random_masked_preds.std(),
            "Mean Δ Prediction (Random)": delta_rand.mean(),
            "Std Δ Prediction (Random)": delta_rand.std(),
            "Paired t-stat": t_stat,
            "p-value": p_val,
            "Num Repeats": n_repeats,
        })

    # Convert to DataFrame
    df_mask = pd.DataFrame(results)
    df_mask = df_mask.sort_values(by="p-value", ascending=True)

    # Streamlit display
    st.markdown(f"### 🔬 Robustness: Random Perturbation of Top {top_k_candidates} Candidates vs Random Residues")
    st.dataframe(df_mask, use_container_width=True)

    st.markdown(
        f"**Original Prediction:** `{original_pred:.4f}`  \n"
        f"**Mean Δ (Top Subset):** `{df_mask['Mean Δ Prediction (Top Subset)'].mean():.4f}` ± `{df_mask['Std Δ Prediction (Top Subset)'].mean():.4f}`  \n"
        f"**Mean Δ (Random):** `{df_mask['Mean Δ Prediction (Random)'].mean():.4f}` ± `{df_mask['Std Δ Prediction (Random)'].mean():.4f}`"
    )

    st.markdown(
        "**Paired t-test** (ΔTop Subset vs ΔRandom): Lower p-value → stronger evidence that masking important residues significantly changes the prediction."
    )    

    explainer.set_mode('eval') 

    return df_mask


def plot_robustness_bar(df_mask):
    """
    Creates a matplotlib figure for Δ Prediction (Top-k vs Random) with error bars.
    df_mask: DataFrame returned by test_robustness_masking
    Returns: matplotlib Figure object
    """
    # Prepare data
    tokens = df_mask["Mask Token"]
    topk_means = df_mask["Mean Δ Prediction (Top Subset)"]
    topk_stds = df_mask["Std Δ Prediction (Top Subset)"]
    rand_means = df_mask["Mean Δ Prediction (Random)"]
    rand_stds = df_mask["Std Δ Prediction (Random)"]
    
    x = np.arange(len(tokens))  # positions
    width = 0.35  # bar width

    fig, ax = get_styled_figure_ax(figsize=(15,6), aspect='none', grid=True)

    # Plot bars
    ax.bar(x - width/2, topk_means, width, yerr=topk_stds, capsize=5, label='Top-k', color=DATASET_COLORS[0])
    ax.bar(x + width/2, rand_means, width, yerr=rand_stds, capsize=5, label='Random', color=DATASET_COLORS[1])

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.set_ylabel('Δ Prediction')
    ax.set_xlabel('Mask Token')
    style_legend(ax, ncol=2, bbox_to_anchor=(0.5, 1.15))
    
    return fig
    


def fraction_of_mass_for_importance(
    per_layer_cams: np.ndarray,
    target_fraction: float = 0.8,
    protein_sequence: str = None,
    display_plot: bool = True
):
    """
    Compute the fraction of input sequence needed to reach a given fraction of total importance.

    Parameters
    ----------
    per_layer_cams : np.ndarray
        Layer-wise CAM / importance maps, shape (n_layers, seq_len)
    target_fraction : float
        Target fraction of total importance to reach (e.g., 0.8 for 80%)
    protein_sequence : str, optional
        Protein sequence (for plotting / reporting)
    display_plot : bool
        Whether to show a cumulative plot in Streamlit

    Returns
    -------
    fraction_needed : float
        Fraction of input residues needed to reach `target_fraction` of total importance
    cumulative_importance : np.ndarray
        Sorted cumulative importance for plotting
    sorted_indices : np.ndarray
        Indices of residues sorted by importance
    """

    # Aggregate importance across layers
    mean_cam = np.mean(per_layer_cams, axis=0).flatten()  # shape (seq_len,)
    seq_len = len(mean_cam)

    # Sort residues by importance descending
    sorted_indices = np.argsort(mean_cam)[::-1]
    sorted_importance = mean_cam[sorted_indices]

    # Compute cumulative sum of importance
    cumulative_importance = np.cumsum(sorted_importance)
    total_importance = cumulative_importance[-1]

    # Normalize to get fraction of total importance
    cumulative_fraction = cumulative_importance / total_importance

    # Find how many residues needed to reach target fraction
    idx_needed = np.searchsorted(cumulative_fraction, target_fraction, side='left') + 1
    fraction_needed = idx_needed / seq_len

    # Streamlit plotting
    if display_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, seq_len + 1) / seq_len, cumulative_fraction, marker='o')
        ax.axhline(target_fraction, color='r', linestyle='--', label=f'{int(target_fraction*100)}% importance')
        ax.axvline(fraction_needed, color='g', linestyle='--', label=f'{fraction_needed:.2f} fraction of residues')
        ax.set_xlabel("Fraction of residues included (sorted by importance)")
        ax.set_ylabel("Cumulative importance fraction")
        ax.set_title("Fraction of Input Needed to Explain Importance")
        ax.legend()
        st.pyplot(fig)

    return fraction_needed, cumulative_fraction, sorted_indices



def fraction_of_mass_per_layer(
    per_layer_cams: np.ndarray,
    target_fraction: float = 0.8,
    protein_sequence: str = None,
    display_plot: bool = True
):
    """
    Compute fraction of input sequence needed to reach target fraction of importance,
    with separate colored curves and shaded variability per layer.

    Parameters
    ----------
    per_layer_cams : np.ndarray
        Layer-wise CAM / importance maps, shape (n_layers, seq_len)
    target_fraction : float
        Target fraction of total importance to reach
    protein_sequence : str, optional
        Protein sequence (for reporting top residues)
    display_plot : bool
        Whether to show a cumulative plot in Streamlit

    Returns
    -------
    fractions_needed : dict
        Fraction of input residues needed per layer + mean
    cumulative_fractions : dict
        Cumulative fractions per layer + mean
    sorted_indices : np.ndarray
        Indices of residues sorted by mean importance
    """

    n_layers, seq_len = per_layer_cams.shape

    # Compute mean CAM
    mean_cam = np.mean(per_layer_cams, axis=0).flatten()
    sorted_indices = np.argsort(mean_cam)[::-1]

    cumulative_fractions = {}
    fractions_needed = {}

    # --- Mean CAM ---
    sorted_mean = mean_cam[sorted_indices]
    total_importance = np.sum(sorted_mean)
    
    if total_importance > 0:
        cum_mean = np.cumsum(sorted_mean) / total_importance
        fractions_needed['mean'] = (np.searchsorted(cum_mean, target_fraction) + 1) / seq_len
    else:
        cum_mean = np.zeros_like(sorted_mean)
        fractions_needed['mean'] = 0.0
        
    cumulative_fractions['mean'] = cum_mean

    # --- Layers ---
    cum_layers = []
    for l in range(n_layers):
        layer_cam = per_layer_cams[l].flatten()
        sorted_layer = layer_cam[sorted_indices]  # align with mean sort
        total_layer_importance = np.sum(sorted_layer)
        
        if total_layer_importance > 0:
            cum_layer = np.cumsum(sorted_layer) / total_layer_importance
            fractions_needed[f'layer_{l}'] = (np.searchsorted(cum_layer, target_fraction) + 1) / seq_len
        else:
            cum_layer = np.zeros_like(sorted_layer)
            fractions_needed[f'layer_{l}'] = 0.0
            
        cumulative_fractions[f'layer_{l}'] = cum_layer
        cum_layers.append(cum_layer)

    cum_layers = np.stack(cum_layers, axis=0)  # shape (n_layers, seq_len)

    # --- Plot ---
    if display_plot:
        # plt.figure(figsize=(8, 5))
        fig, ax = get_styled_figure_ax(figsize=(16, 10), aspect='none')
        colors = ['#EA805D', '#6a408d', '#7E7EA5', '#95BB63', '#487C7D','#4897DC']

        for l in range(n_layers):
            plt.plot(np.arange(1, seq_len + 1) / seq_len, cum_layers[l],
                     # color=colors(l), 
                     color=colors[l], 
                     lw=2, alpha=0.7, label=f'Layer {l}')

        # Mean CAM on top
        plt.plot(np.arange(1, seq_len + 1) / seq_len, cum_mean, lw=3, color='black', label='Mean CAM')

        # Target fraction line
        plt.axhline(target_fraction, color='r', linestyle='--', label=f'{int(target_fraction*100)}% importance')

        plt.xlabel("Fraction of residues included (sorted by mean importance)")
        plt.ylabel("Cumulative importance fraction")
        # plt.title("Cumulative Importance per Layer with Colored Variability")
        # plt.legend()
        style_legend(ax, ncol=4, bbox_to_anchor=(0.5, 1.15))
        plt.savefig('results/app/figs/cumulative_importance_per_layer.svg', bbox_inches='tight')
        st.pyplot(plt.gcf())


    return fractions_needed, cumulative_fractions, sorted_indices


def fraction_of_mass_sliding_window(
    per_layer_cams: np.ndarray,
    window_size: int = 5,
    stride: int = 1,
    target_fraction: float = 0.8,
    protein_sequence: str = None,
    display_plot: bool = True
):
    """
    Compute fraction of input mass needed and most important regions using sliding windows.

    Parameters
    ----------
    per_layer_cams : np.ndarray
        Layer-wise CAM / importance maps, shape (n_layers, seq_len)
    window_size : int
        Size of sliding window (number of residues)
    stride : int
        Stride for sliding window
    target_fraction : float
        Target fraction of total importance to reach
    protein_sequence : str, optional
        Protein sequence (for plotting / reporting)
    display_plot : bool
        Whether to display cumulative importance plot in Streamlit

    Returns
    -------
    fraction_needed : float
        Fraction of protein sequence needed to reach target fraction
    top_windows : list of tuples
        List of most important windows as (start_idx, end_idx, importance_fraction)
    cumulative_fraction : np.ndarray
        Cumulative importance fraction over sorted windows
    sorted_windows : np.ndarray
        Indices of windows sorted by importance
    """

    n_layers, seq_len = per_layer_cams.shape

    # Aggregate CAMs across layers
    mean_cam = np.mean(per_layer_cams, axis=0).flatten()

    # Compute sliding window sums
    window_sums = []
    window_indices = []
    for start in range(0, seq_len - window_size + 1, stride):
        window_sum = mean_cam[start:start+window_size].sum()
        window_sums.append(window_sum)
        window_indices.append((start, start+window_size))

    window_sums = np.array(window_sums)
    window_indices = np.array(window_indices)

    # Sort windows by importance descending
    sorted_idx = np.argsort(window_sums)[::-1]
    sorted_sums = window_sums[sorted_idx]
    sorted_windows = window_indices[sorted_idx]

    # Cumulative fraction of total importance
    cumulative_fraction = np.cumsum(sorted_sums) / np.sum(mean_cam)

    # Find number of windows needed to reach target fraction
    idx_needed = np.searchsorted(cumulative_fraction, target_fraction) + 1
    # Convert to fraction of residues (overlapping windows counted once)
    covered_positions = set()
    for w in sorted_windows[:idx_needed]:
        covered_positions.update(range(w[0], w[1]))
    fraction_needed = len(covered_positions) / seq_len

    # Collect top windows info
    top_windows = [(w[0], w[1], sorted_sums[i]/np.sum(mean_cam)) for i, w in enumerate(sorted_windows[:idx_needed])]

    # Plot
    if display_plot:
        plt.figure(figsize=(7,4))
        plt.plot(np.arange(1, len(sorted_sums)+1), cumulative_fraction, marker='o')
        plt.axhline(target_fraction, color='r', linestyle='--', label=f'{int(target_fraction*100)}% importance')
        plt.xlabel("Number of windows (sorted by importance)")
        plt.ylabel("Cumulative importance fraction")
        plt.title(f"Cumulative Importance using Sliding Windows (size={window_size})")
        plt.legend()
        st.pyplot(plt.gcf())

    return fraction_needed, top_windows, cumulative_fraction, sorted_windows


def plot_sliding_window_heatmap(
    protein_length: int,
    top_windows: list,
    protein_sequence: str = None,
    max_per_row: int = 50,
    title: str = "Protein Sliding-Window Heatmap"
):
    """
    Plot a heatmap of protein importance with multiple rows for long sequences.

    Parameters
    ----------
    protein_length : int
        Length of protein sequence
    top_windows : list of tuples
        Sliding windows as (start_idx, end_idx, fraction_of_total_importance)
    protein_sequence : str, optional
        Protein sequence to overlay on heatmap
    max_per_row : int
        Max number of residues per row
    title : str
        Plot title
    """

    # Number of rows needed
    total_rows = int(np.ceil(protein_length / max_per_row))
    z = np.zeros((total_rows, max_per_row))
    text = [[''] * max_per_row for _ in range(total_rows)]

    # Map residue importance from windows
    residue_importance = np.zeros(protein_length)
    for start, end, frac in top_windows:
        residue_importance[start:end] += frac

    # Normalize
    residue_importance /= residue_importance.max() if residue_importance.max() > 0 else 1.0

    # Fill z and text arrays
    for idx in range(protein_length):
        row = idx // max_per_row
        col = idx % max_per_row
        z[row, col] = residue_importance[idx]
        if protein_sequence is not None:
            text[row][col] = protein_sequence[idx]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z[::-1],  # flip rows to have top-down
        text=text[::-1],
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"},
        colorscale='Reds',
        showscale=True,
        xgap=1,
        ygap=1,
        zmin=0,
        zmax=1
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Residue Position",
        yaxis=dict(showticklabels=False),
        height=total_rows*20 + 50 
    )

    st.plotly_chart(fig, use_container_width=True)
