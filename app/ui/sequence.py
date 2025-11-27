import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from MGraphDTA.visualization.preprocessing import VOCAB_PROTEIN
from xai.visualization.sequence_visual import plot_sequence_attention, plot_sequence_attention_multi_layer

# Constants
INT2VOCAB = {str(v): k for k, v in VOCAB_PROTEIN.items()}
INT2VOCAB['0'] = '-'

def analyze_top_regions(grayscale_cam, protein_sequence_aa, top_k=5):
    cam_1d = grayscale_cam.flatten()
    top_indices = np.argsort(cam_1d)[-top_k:][::-1]
    
    message = f"Top {top_k} most important positions:\n"
    for rank, idx in enumerate(top_indices, 1):
        if idx < len(protein_sequence_aa):
            aa = protein_sequence_aa[idx]
            weight = cam_1d[idx]
            if np.isclose(weight, 0.0):
                continue
            message += f"{rank}. Position {idx:04d}: {aa} (attention: {weight:.4f})\n"
    
    return message


def create_attention_visualization(grayscale_cam_list, layer_names, protein_sequence,
                        max_per_row=60):
    """Create the visualization figure and return it using Plotly"""

    # Normalize CAMs
    if isinstance(grayscale_cam_list, (np.ndarray, torch.Tensor)) and not isinstance(grayscale_cam_list, list):
        arr = grayscale_cam_list
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr)
        cams = [arr[i] for i in range(arr.shape[0])] if arr.ndim == 3 else [arr]
    elif isinstance(grayscale_cam_list, list):
        cams = [c.cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c) for c in grayscale_cam_list]
    else:
        cams = [np.asarray(grayscale_cam_list)]

    # Convert sequence
    if isinstance(protein_sequence, torch.Tensor):
        seq_np = protein_sequence.detach().cpu().flatten().numpy()
    else:
        seq_np = np.asarray(protein_sequence).flatten()

    seq_len = len(seq_np)
    seq_letters_full = [INT2VOCAB.get(str(int(k)), '-') for k in seq_np]

    # Process CAMs 
    def cam_to_1d(cam):
        cam = np.asarray(cam)
        if cam.ndim == 1:
            return cam
        elif cam.ndim == 2:
            return cam.flatten()
        elif cam.ndim == 3:
            return cam.mean(axis=0).flatten()
        return cam.flatten()

    cam_1d_list = []
    for cam in cams:
        cam1 = cam_to_1d(cam)
        if cam1.size > seq_len:
            cam1 = cam1[:seq_len]
        elif cam1.size < seq_len:
            cam1 = np.pad(cam1, (0, seq_len - cam1.size), mode='constant')
        cam_1d_list.append(cam1)

    # Create figure with Plotly
    n_cols = len(cam_1d_list)
    n_rows = int(np.ceil(seq_len / max_per_row))

    # Create subplots
    fig = make_subplots(rows=2, cols=n_cols, subplot_titles=[f'{name}\nAttention Along Sequence' for name in layer_names])

    for col_idx, cam1_full in enumerate(cam_1d_list):
        # Trim trailing zeros
        threshold = 1e-4
        indices = np.flatnonzero(cam1_full > threshold)
        if indices.size > 0:
            last_index = min(len(cam1_full)-1, indices[-1]+10)
            last_index = max(len("".join(seq_letters_full[:last_index+1]).strip('-'))+10, last_index)
        else:
            last_index = min(len("".join(seq_letters_full).strip('-'))+10, len(cam1_full)-1)

        cam1 = cam1_full[:last_index + 1]
        positions = np.arange(last_index+1)
        seq_letters = seq_letters_full[:last_index+1]

        # Add line plot
        fig.add_trace(go.Scatter(x=positions, y=cam1, mode='lines', line=dict(width=1.5), opacity=0.8), row=1, col=col_idx+1)

        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=[cam1], 
            colorscale='Reds',
            colorbar=dict(
                len=0.5,            # fraction of figure height
                y=0.125               # vertical center of colorbar
            ),
        ),row=2, col=col_idx+1)

    # Update layout
    fig_height = max(4, .5 * n_rows) 
    fig_width = max(10, 7 * n_cols)   
    fig.update_layout(height=fig_height*50, width=fig_width*50, title_text="Attention across Protein Sequence")
    fig.update_xaxes(title_text='Sequence Position', row=1, col=1)
    fig.update_yaxes(title_text='Attention Weight', row=1, col=1)

    figs_sequence_heatmap = []
    for cam_1d, layer_name in zip(cam_1d_list, layer_names):
        figs_sequence_heatmap.append(
            plot_sequence_attention(
                seq_letters_full, 
                cam_1d, 
                max_per_row, 
                title=f"{layer_name} Protein Sequence with Attention"))

    # Create multi-layer attention visualization
    fig_multi_layer = plot_sequence_attention_multi_layer(
        seq_letters_full,
        cam_1d_list,
        layer_names,
        max_per_row,
        title="Multi-Layer Attention Visualization"
    )

    out = {
        'seq_letters': seq_letters_full,
        'figattention_lineplot': fig,
        'figs_sequence_heatmap': figs_sequence_heatmap,
        'fig_multi_layer_attention': fig_multi_layer,
        'per_layer_cam_1d': cam_1d_list
    }

    return out
