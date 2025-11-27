import numpy as np
import plotly.graph_objects as go
import re
from lime.explanation import Explanation
import matplotlib.pyplot as plt
from src.utils.plot_utils import set_plot_style

def plot_sequence_attention(seq_letters, attention_weights, max_per_row=60, title="Protein Sequence with Attention"):
    """
    Plot the protein sequence with attention weights as colored background using Matplotlib.
    
    seq_letters: list of amino acid letters (strings)
    attention_weights: list or np.array of attention values (same length as seq_letters)
    max_per_row: number of amino acids per row
    """
    set_plot_style()
    
    # Filter out trailing padding ('-')
    last_char_idx = -1
    for i in range(len(seq_letters) - 1, -1, -1):
        if seq_letters[i] != '-':
            last_char_idx = i
            break
    
    if last_char_idx == -1: # All padding or empty
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No sequence data to display.", ha='center', va='center')
        return fig

    # Slice to the actual sequence length
    seq_letters = seq_letters[:last_char_idx + 1]
    attention_weights = attention_weights[:last_char_idx + 1]

    # Normalize weights to [0,1]
    weights = np.array(attention_weights)
    if weights.max() - weights.min() == 0:
        norm_weights = np.zeros_like(weights)
    else:
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Determine number of rows
    seq_len = len(seq_letters)
    n_rows = int(np.ceil(seq_len / max_per_row))
    
    # Create 2D grid for heatmap and text
    z = np.full((n_rows, max_per_row), np.nan) # Use NaN for empty cells
    
    fig, ax = plt.subplots(figsize=(max_per_row * 0.4, n_rows * 0.8)) # Thinner rows
    
    # Fill data for heatmap and add text
    for idx, (letter, weight) in enumerate(zip(seq_letters, norm_weights)):
        row = idx // max_per_row
        col = idx % max_per_row
        
        display_letter = letter
        if display_letter == 'PAD':
            display_letter = '-'
        
        z[row, col] = weight
        ax.text(col, row, display_letter, 
                ha='center', va='center', color='black', fontsize=9)
    
    # Plot heatmap
    cmap = plt.cm.get_cmap('Reds')
    cmap.set_bad(color='white') # Make NaN values white
    im = ax.imshow(z, cmap=cmap, aspect='equal', interpolation='nearest', vmin=0, vmax=1)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.01, shrink=0.4, aspect=20)
    # cbar.set_label("Normalized Attention Weight")
    cbar.set_label("")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(0, max_per_row, max(1, max_per_row // 10)))
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f'Row {i+1}' for i in range(n_rows)], fontsize=10)
    
    # ax.set_title(f"{title}", fontsize=14)
    ax.set_xlabel("Sequence Position in Row")
    ax.set_ylabel("Row")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    fig.tight_layout()
    return fig

def plot_sequence_attention_enhanced(seq_letters, attention_weights, max_per_row=60, title="Protein Sequence with Attention (Enhanced Contrast)"):
    """
    Plots sequence attention with enhanced contrast for lower values.
    """
    set_plot_style()
    
    # Filter out trailing padding ('-')
    last_char_idx = -1
    for i in range(len(seq_letters) - 1, -1, -1):
        if seq_letters[i] != '-':
            last_char_idx = i
            break

    
    if last_char_idx == -1: # All padding or empty
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No sequence data to display.", ha='center', va='center')
        return fig

    # Slice to the actual sequence length
    seq_letters = seq_letters[:last_char_idx + 1]
    attention_weights = attention_weights[:last_char_idx + 1]

    # Normalize weights to [0,1]
    weights = np.array(attention_weights)
    if weights.max() - weights.min() == 0:
        norm_weights = np.zeros_like(weights)
    else:
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # --- Apply power-law scaling (sqrt) ---
    norm_weights = np.power(norm_weights, 0.5)

    # Determine number of rows
    seq_len = len(seq_letters)
    n_rows = int(np.ceil(seq_len / max_per_row))
    
    # Create 2D grid for heatmap and text
    z = np.full((n_rows, max_per_row), np.nan) # Use NaN for empty cells
    
    fig, ax = plt.subplots(figsize=(max_per_row * 0.4, n_rows * 0.8)) # Thinner rows
    
    for idx, (letter, weight) in enumerate(zip(seq_letters, norm_weights)):
        row = idx // max_per_row
        col = idx % max_per_row
        
        display_letter = letter
        if display_letter == 'PAD':
            display_letter = '-'
        
        z[row, col] = weight
        ax.text(col, row, display_letter, 
                ha='center', va='center', color='black', fontsize=9)
    
    # Plot heatmap
    cmap = plt.cm.get_cmap('Reds')
    cmap.set_bad(color='white') # Make NaN values white
    im = ax.imshow(z, cmap=cmap, aspect='equal', interpolation='nearest', vmin=0, vmax=1)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.01, shrink=0.5, aspect=20)
    cbar.set_label("Enhanced Attention Weight (sqrt-scaled)")
    
    ax.set_xticks(np.arange(0, max_per_row, max(1, max_per_row // 10)))
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f'Row {i+1}' for i in range(n_rows)], fontsize=10)
    
    ax.set_xlabel("Sequence Position in Row")
    ax.set_ylabel("Row")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    fig.tight_layout()
    return fig

def plot_sequence_attention_multi_layer(
    seq_letters, attention_weights_list, layer_names, max_per_row=60, 
    title="Protein Sequence with Multi-Layer Attention"):
    """
    Plot the protein sequence with attention weights from all layers.
    
    Args:
        seq_letters: list of amino acid letters (strings)
        attention_weights_list: list of attention arrays, one per layer
        layer_names: list of layer names
        max_per_row: number of amino acids per row
        title: title for the plot
    
    Returns:
        Plotly figure with multi-layer attention visualization
    """
    last_char_idx = -1
    for i in range(len(seq_letters) - 1, -1, -1):
        if seq_letters[i] != '-':
            last_char_idx = i
            break

    
    if last_char_idx == -1: # All padding or empty
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No sequence data to display.", ha='center', va='center')
        return fig

    # Slice to the actual sequence length
    seq_letters = seq_letters[:last_char_idx + 1]
    attention_weights_list = [ aw[:last_char_idx + 1] for aw in attention_weights_list ]
    
    # Determine number of sequence rows
    seq_len = len(seq_letters)
    n_seq_rows = int(np.ceil(seq_len / max_per_row))
    n_layers = len(attention_weights_list)
    
    # Total number of rows: n_seq_rows * (n_layers + 1)
    # For each sequence row, we have n_layers attention rows + 1 sequence row
    total_rows = n_seq_rows * (n_layers + 1)
    
    # Create 2D grid for heatmap
    z = np.zeros((total_rows, max_per_row))
    text = np.full((total_rows, max_per_row), '', dtype=object)
    
    # Process each sequence row
    for seq_row in range(n_seq_rows):
        start_idx = seq_row * max_per_row
        end_idx = min(start_idx + max_per_row, seq_len)
        
        # For each layer, add attention row
        for layer_idx, (attention_weights, layer_name) in enumerate(zip(attention_weights_list, layer_names)):
            # Normalize weights to [0,1]
            weights = np.array(attention_weights)
            norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            
            # Calculate the row index for this layer's attention
            attention_row = seq_row * (n_layers + 1) + layer_idx
            
            # Fill attention data for this row
            for pos in range(max_per_row):
                seq_idx = start_idx + pos
                if seq_idx < end_idx:
                    z[attention_row, pos] = norm_weights[seq_idx]
                    # Add layer name as text for the first position
                    # if pos == 0:
                    #     text[attention_row, pos] = f"{layer_name}"
                    # else:
                    text[attention_row, pos] = ""
                else:
                    z[attention_row, pos] = 0
                    text[attention_row, pos] = ""
        
        # Add the actual sequence row
        sequence_row = seq_row * (n_layers + 1) + n_layers
        
        for pos in range(max_per_row):
            seq_idx = start_idx + pos
            if seq_idx < end_idx:
                letter = seq_letters[seq_idx]
                # Map special tokens
                display_letter = letter
                if display_letter == 'PAD':
                    display_letter = '-'
                
                z[sequence_row, pos] = 0  # Neutral background for sequence
                text[sequence_row, pos] = display_letter
            else:
                z[sequence_row, pos] = 0
                text[sequence_row, pos] = ""
    
    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z[::-1],  # flip rows to have top-down
        text=text[::-1],
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"},
        colorscale='Reds',
        showscale=True,
        xgap=1, 
        ygap=1,  # grid lines between cells
        zmin=0,
        zmax=1
    ))
    
    # Create custom y-axis labels
    y_tickvals = []
    y_ticktext = []

    for seq_row in range(n_seq_rows):
        # Add tick for each layer attention row
        for layer_idx in range(n_layers):
            attention_row = seq_row * (n_layers + 1) + layer_idx
            y_tickvals.append(total_rows - 1 - attention_row)  # Account for flip
            y_ticktext.append(f"{layer_names[layer_idx]}")
        
        # Add tick for sequence row
        sequence_row = seq_row * (n_layers + 1) + n_layers
        y_tickvals.append(total_rows - 1 - sequence_row)  # Account for flip
        y_ticktext.append(f"Sequence {seq_row + 1}")
    
    fig.update_layout(
        title=f"{title} (Multi-Layer Attention, {max_per_row} per row)",
        xaxis_title="Sequence Position",
        yaxis_title="Layer & Sequence",
        yaxis=dict(
            tickmode='array', 
            tickvals=y_tickvals,
            ticktext=y_ticktext
        ),
        xaxis=dict(tickmode='array', tickvals=list(range(0, max_per_row, max(1, max_per_row // 10)))),
        height=max(400, total_rows * 25)  # Adjust height based on number of rows
    )
    
    return fig


def generate_sequence_html(protein_sequence: str, explanation: Explanation) -> str:
    """
    Generates an HTML string to visualize LIME weights on a protein sequence.

    Args:
        protein_sequence: The original amino acid sequence string.
        explanation: The LIME explanation object.

    Returns:
        An HTML string with colored amino acids.
    """
    explanation_list = explanation.as_list()
    
    # 1. Create a weight map for each position in the sequence
    weights = np.zeros(len(protein_sequence))
    for feature, weight in explanation_list:
        # LIME features are like 'pos_152 > 14.00', we just need the position
        match = re.search(r'pos_(\d+)', feature)
        if match:
            pos = int(match.group(1))
            if pos < len(weights):
                print(f"Feature: {feature}, Position: {pos}, Weight: {weight}")
                weights[pos] += weight # Add weight (can be multiple bins for one pos)

    # 2. Normalize weights to be between -1 and 1 for coloring
    max_abs_weight = np.max(np.abs(weights))
    if max_abs_weight == 0:
        max_abs_weight = 1 # Avoid division by zero
    
    norm_weights = weights / max_abs_weight

    # 3. Build the HTML string
    html = '<div style="font-family: monospace; font-size: 14px; line-height: 20px; white-space: pre-wrap; word-wrap: break-word;">'
    for i, char in enumerate(protein_sequence):
        weight = norm_weights[i]
        if weight > 0: # Positive contribution (stronger binding) -> Green
            alpha = weight
            color = f'rgba(40, 167, 69, {alpha})' # Green
        elif weight < 0: # Negative contribution (weaker binding) -> Red
            alpha = -weight
            color = f'rgba(220, 53, 69, {alpha})' # Red
        else: # No contribution
            color = 'rgba(240, 240, 240, 0.5)' # Light grey

        # Add a tooltip to show the raw weight on hover
        tooltip = f'Position: {i}\nWeight: {weights[i]:.4f}'
        html += f'<span style="background-color: {color};" title="{tooltip}">{char}</span>'
        
        # Add a line break every 80 characters for readability
        if (i + 1) % 80 == 0:
            html += '<br>'

    html += '</div>'
    return html

