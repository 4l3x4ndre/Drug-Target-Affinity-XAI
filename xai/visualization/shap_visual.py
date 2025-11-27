import numpy as np
import matplotlib.pyplot as plt
from src.utils.plot_utils import set_plot_style, style_legend
DATASET_COLORS = ['#95BB63', '#BCBCE0', '#77b5b6', '#EA805D']


def plot_shap_explanation_plots(shap_results, max_per_row=60, title="SHAP Explanation for Protein Sequence", show_sequence=True):
    """
    Plot SHAP values for protein sequence using matplotlib, returning separate figures.
    
    Args:
        shap_results: Dictionary from explain_shap method containing SHAP values and metadata
        max_per_row: Maximum number of amino acids to display per row
        title: Title for the plot
        show_sequence: Whether to show the sequence heatmap
        
    Returns:
        A dictionary of matplotlib figure objects.
    """
    set_plot_style()
    
    shap_values = shap_results['shap_values']
    amino_acid_labels = shap_results['amino_acid_labels']
    original_prediction = shap_results['original_prediction']
    method = shap_results.get('method', 'unknown')
    
    shap_values_flat = shap_values.flatten()
    
    # Filter out padding values
    non_padding_indices = [i for i, label in enumerate(amino_acid_labels) if label != '-']
    shap_values_actual = shap_values_flat[non_padding_indices]
    amino_acids_actual = [amino_acid_labels[i] for i in non_padding_indices]
    actual_length = len(amino_acids_actual)

    figs = {}

    # Figure 1: SHAP values as a heatmap
    if show_sequence:
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(20, 10))
        
        n_rows = (actual_length + max_per_row - 1) // max_per_row
        shap_matrix = np.full((n_rows, max_per_row), np.nan)
        
        for i in range(actual_length):
            row, col = divmod(i, max_per_row)
            shap_matrix[row, col] = shap_values_actual[i]
            
        im = ax_heatmap.imshow(shap_matrix, cmap='RdBu', aspect='auto', interpolation='nearest')
        fig_heatmap.colorbar(im, ax=ax_heatmap, label="SHAP Value")
        ax_heatmap.set_title('SHAP Values by Position', fontsize=20)
        ax_heatmap.set_xlabel('Position in Row')
        ax_heatmap.set_ylabel('Sequence Row')

        for i in range(actual_length):
            row, col = divmod(i, max_per_row)
            ax_heatmap.text(col, row, amino_acids_actual[i], ha='center', va='center', color='white', fontsize=8)
        
        fig_heatmap.suptitle(f'{title} - {method.upper()} Method (Prediction: {original_prediction:.4f})', fontsize=24)
        fig_heatmap.tight_layout(rect=[0, 0, 1, 0.96])
        figs['heatmap'] = fig_heatmap

    # Figure 2: SHAP values as a bar plot
    fig_bar, ax_bar = plt.subplots(figsize=(20, 8))
    positions = np.arange(actual_length)
    colors = [DATASET_COLORS[-1] if x < 0 else DATASET_COLORS[0] for x in shap_values_actual]
    ax_bar.bar(positions, shap_values_actual, color=colors, alpha=0.8)
    ax_bar.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_bar.set_title('SHAP Values for All Positions', fontsize=20)
    ax_bar.set_xlabel('Sequence Position')
    ax_bar.set_ylabel('SHAP Value')
    ax_bar.grid(True, which='major', linestyle='--', alpha=0.6)
    fig_bar.tight_layout()
    figs['bar'] = fig_bar

    # Figure 3: Top contributing positions
    fig_top, ax_top = plt.subplots(figsize=(15, 8))

    # Filter for non-zero SHAP values before selecting top contributors
    non_zero_mask = shap_values_actual != 0
    shap_values_non_zero = shap_values_actual[non_zero_mask]
    
    # Get original indices of non-zero values to map back later
    original_indices = np.where(non_zero_mask)[0]

    if len(shap_values_non_zero) > 0:
        abs_values = np.abs(shap_values_non_zero)
        
        # Determine how many top features to show (up to 15)
        num_top_features = min(15, len(shap_values_non_zero))
        
        # Get indices of top features within the non-zero array
        top_indices_in_non_zero = np.argsort(abs_values)[-num_top_features:][::-1]
        
        # Map these indices back to their original positions in shap_values_actual
        top_original_indices = original_indices[top_indices_in_non_zero]

        top_labels = [f"Pos {non_padding_indices[i]}\n({amino_acids_actual[i]})" for i in top_original_indices]
        top_values = shap_values_actual[top_original_indices]
        top_colors = [DATASET_COLORS[-1] if x < 0 else DATASET_COLORS[0] for x in top_values]
        
        ax_top.bar(np.arange(len(top_labels)), top_values, color=top_colors, alpha=0.8)
        ax_top.set_xticks(np.arange(len(top_labels)))
        ax_top.set_xticklabels(top_labels, rotation=45, ha='right')
    else:
        ax_top.text(0.5, 0.5, "No non-zero contributing positions found.", ha='center', va='center')

    ax_top.axhline(0, color='black', linestyle='--', alpha=0.5)
    # ax_top.set_title('Top Contributing Positions (Non-Zero)', fontsize=20)
    ax_top.set_ylabel('SHAP Value')
    ax_top.grid(True, which='major', linestyle='--', alpha=0.6)
    fig_top.tight_layout()
    figs['top_contrib'] = fig_top
    
    return figs


def plot_shap_waterfall(shap_results, max_features=20):
    """
    Create a correct SHAP waterfall plot using matplotlib.
    
    Args:
        shap_results: Dictionary from explain_shap method
        max_features: Maximum number of features to display
        
    Returns:
        matplotlib figure object
    """
    set_plot_style()

    shap_values = shap_results['shap_values'].flatten()
    base_value = shap_results['base_value']
    amino_acid_labels = shap_results['amino_acid_labels']
    
    # Filter out padding values
    non_padding_indices = [i for i, label in enumerate(amino_acid_labels) if label != '-']
    shap_values_actual = shap_values[non_padding_indices]
    amino_acids_actual = [amino_acid_labels[i] for i in non_padding_indices]
    
    # Further filter for non-zero SHAP values
    non_zero_indices = np.where(shap_values_actual != 0)[0]
    shap_values_non_zero = shap_values_actual[non_zero_indices]
    feature_names_non_zero = [f"Pos {non_padding_indices[i]}: {amino_acids_actual[i]}" for i in non_zero_indices]

    if len(shap_values_non_zero) == 0:
        # Handle case with no contributing features
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No feature contributions to display.", ha='center', va='center')
        return fig

    # Sort by absolute SHAP value and select top features
    sorted_indices = np.argsort(np.abs(shap_values_non_zero))[::-1]
    
    if len(sorted_indices) > max_features:
        top_indices = sorted_indices[:max_features]
        other_shap_sum = shap_values_non_zero[sorted_indices[max_features:]].sum()
        top_shap_values = shap_values_non_zero[top_indices]
        top_features = [feature_names_non_zero[i] for i in top_indices]
        
        if other_shap_sum != 0:
            top_shap_values = np.append(top_shap_values, other_shap_sum)
            top_features.append(f'Other {len(shap_values_non_zero) - max_features} features')
    else:
        top_shap_values = shap_values_non_zero
        top_features = feature_names_non_zero

    # Order by position in the plot (descending by SHAP value)
    order = np.argsort(-top_shap_values)
    top_shap_values = top_shap_values[order]
    top_features = [top_features[i] for i in order]

    # --- Matplotlib Waterfall ---
    fig, ax = plt.subplots(figsize=(13, len(top_features) * 1))
    
    y_pos = np.arange(len(top_features))
    
    cumulative = base_value
    
    # Plot bars feature by feature
    for i, (value, feature) in enumerate(zip(top_shap_values, top_features)):
        color = DATASET_COLORS[0] if value > 0 else DATASET_COLORS[-1]
        ax.barh(y_pos[len(top_features) - 1 - i], value, left=cumulative, color=color)
        cumulative += value

    # Add vertical lines for base and final prediction
    ax.axvline(base_value, color='gray', linestyle='--', linewidth=1.5, label=f'Base Value = {base_value:.4f}', alpha=0.8)
    ax.axvline(cumulative, color=DATASET_COLORS[2], linestyle='--', linewidth=1.5, label=f'Final Prediction = {cumulative:.4f}', alpha=0.8)
    
    # Adjust x-limits
    plot_min = min(ax.get_xlim()[0], base_value, cumulative)
    plot_max = max(ax.get_xlim()[1], base_value, cumulative)
    padding = (plot_max - plot_min) * 0.05
    ax.set_xlim(plot_min - padding, plot_max + padding)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("SHAP Value (Contribution to Prediction)")
    # ax.set_title("SHAP Waterfall Plot")
    # ax.legend()
    style_legend(ax, ncol=2, bbox_to_anchor=(0.5, 1.215))
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    return fig


def create_shap_summary_cards(shap_results):
    """
    Create summary cards for SHAP results that can be displayed in Streamlit.
    
    Args:
        shap_results: Dictionary from explain_shap method
        
    Returns:
        Dictionary with summary statistics
    """
    shap_values = shap_results['shap_values']
    amino_acid_labels = shap_results['amino_acid_labels']
    original_prediction = shap_results['original_prediction']
    method = shap_results.get('method', 'unknown')
    
    # Flatten and get actual sequence
    shap_values_flat = shap_values.flatten()
    
    # Find actual sequence length
    actual_length = 0
    for i, label in enumerate(amino_acid_labels):
        if label != '-':
            actual_length = i + 1
    
    # Get actual values (excluding padding)
    shap_values_actual = shap_values_flat[:actual_length]
    
    # Calculate summary statistics
    positive_contributions = shap_values_actual[shap_values_actual > 0]
    negative_contributions = shap_values_actual[shap_values_actual < 0]
    
    summary = {
        'method': method.upper(),
        'prediction': original_prediction,
        'sequence_length': actual_length,
        'total_positive_contributions': len(positive_contributions),
        'total_negative_contributions': len(negative_contributions),
        'max_positive_contribution': float(np.max(positive_contributions)) if len(positive_contributions) > 0 else 0.0,
        'max_negative_contribution': float(np.min(negative_contributions)) if len(negative_contributions) > 0 else 0.0,
        'mean_absolute_contribution': float(np.mean(np.abs(shap_values_actual))),
        'std_contribution': float(np.std(shap_values_actual)),
        'top_positive_position': int(np.argmax(shap_values_actual)),
        'top_negative_position': int(np.argmin(shap_values_actual)),
        'top_positive_aa': amino_acid_labels[np.argmax(shap_values_actual)],
        'top_negative_aa': amino_acid_labels[np.argmin(shap_values_actual)]
    }
    
    return summary


def display_shap_summary_streamlit(shap_results):
    """
    Display SHAP summary in Streamlit format.
    
    Args:
        shap_results: Dictionary from explain_shap method
    """
    import streamlit as st
    
    summary = create_shap_summary_cards(shap_results)
    
    # Create columns for summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Method",
            value=summary['method'],
            help="SHAP explanation method used"
        )
    
    with col2:
        st.metric(
            label="Prediction",
            value=f"{summary['prediction']:.4f}",
            help="Model prediction value"
        )
    
    with col3:
        st.metric(
            label="Sequence Length",
            value=summary['sequence_length'],
            help="Length of protein sequence (excluding padding)"
        )
    
    with col4:
        st.metric(
            label="Mean |SHAP|",
            value=f"{summary['mean_absolute_contribution']:.4f}",
            help="Mean absolute SHAP contribution"
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="Max Positive",
            value=f"{summary['max_positive_contribution']:.4f}",
            help=f"Position {summary['top_positive_position']} ({summary['top_positive_aa']})"
        )
    
    with col6:
        st.metric(
            label="Max Negative",
            value=f"{summary['max_negative_contribution']:.4f}",
            help=f"Position {summary['top_negative_position']} ({summary['top_negative_aa']})"
        )
    
    with col7:
        st.metric(
            label="Positive Positions",
            value=summary['total_positive_contributions'],
            help="Number of positions with positive SHAP values"
        )
    
    with col8:
        st.metric(
            label="Negative Positions",
            value=summary['total_negative_contributions'],
            help="Number of positions with negative SHAP values"
        )

