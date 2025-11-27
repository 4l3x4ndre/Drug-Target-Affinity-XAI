import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os
import pandas as pd

from MGraphDTA.visualization.visualization_mgnn import clourMol
from xai.uniprot_to_3D import uniprot_to_html
from ui.sequence import create_attention_visualization, analyze_top_regions
from xai.explainer_robustness import test_robustness_masking, plot_robustness_bar, fraction_of_mass_per_layer, fraction_of_mass_sliding_window, plot_sliding_window_heatmap
from xai.visualization.sequence_visual import plot_sequence_attention, plot_sequence_attention_enhanced
from ui.gnn import gnn_main_visualisation_st, render_gnn_explanation_streamlit
from xai.visualization.gnn_visual import get_gnn_explanation_figures
from ui.interaction import render_interaction_explanations
from xai.visualization.shap_visual import display_shap_summary_streamlit, plot_shap_explanation_plots, plot_shap_waterfall
from src.utils.plot_utils import get_styled_figure_ax

# Define paths for cache files
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
MAIN_CACHE_FILE = os.path.join(CACHE_DIR, "explanation_results.pt")
ROBUSTNESS_CACHE_FILE = os.path.join(CACHE_DIR, "robustness_masking.pt")
GNN_DATA_CACHE_FILE = os.path.join(CACHE_DIR, "gnn_explanation_data.pt")
GNN_OUTPUT_CACHE_FILE = os.path.join(CACHE_DIR, "gnn_output.pt")
LIGAND_IMG_CACHE_FILE = os.path.join(CACHE_DIR, "ligand_img.pt")
SHAP_CACHE_FILE = os.path.join(CACHE_DIR, "shap_results.pt")
LIME_CACHE_FILE = os.path.join(CACHE_DIR, "lime_output.pt")

DATASET_COLORS = ['#95BB63', '#BCBCE0', '#77b5b6', '#EA805D']

def ensure_main_explanation(explainer, smiles_input, protein_input, sample_idx):
    """Ensures the main explanation is generated and cached."""
    if not os.path.exists(MAIN_CACHE_FILE):
        with st.spinner("Generating base explanations... This may take a moment."):
            try:
                output_dict = explainer.explain(smiles_input, protein_input, sample_idx)
                data_to_cache = {
                    'output': output_dict,
                    'smiles': smiles_input,
                    'protein': protein_input,
                    'sample_idx': sample_idx
                }
                torch.save(data_to_cache, MAIN_CACHE_FILE)
                return data_to_cache
            except Exception as e:
                st.error(f"Error generating explanations: {e}")
                return None
    else:
        # Check if cache matches current inputs
        data = torch.load(MAIN_CACHE_FILE, weights_only=False)
        if data['smiles'] != smiles_input or data['protein'] != protein_input:
            # Re-generate if inputs changed
            with st.spinner("Inputs changed. Regenerating base explanations..."):
                try:
                    output_dict = explainer.explain(smiles_input, protein_input, sample_idx)
                    data_to_cache = {
                        'output': output_dict,
                        'smiles': smiles_input,
                        'protein': protein_input,
                        'sample_idx': sample_idx
                    }
                    torch.save(data_to_cache, MAIN_CACHE_FILE)
                    return data_to_cache
                except Exception as e:
                    st.error(f"Error generating explanations: {e}")
                    return None
        return data

def render_attention_analysis(explainer, smiles_input, protein_input, sample_idx):
    st.header("🔬 Attention Analysis")
    
    data = ensure_main_explanation(explainer, smiles_input, protein_input, sample_idx)
    if not data: return

    output_dict = data['output']
    prediction = output_dict['prediction']
    per_layer_cams = output_dict['per_layer_cams']
    atom_att = output_dict['atom_att']
    mol = output_dict['mol_obj']
    layer_names = output_dict['layer_names']
    protein_sequence = output_dict['protein_sequence']

    with st.sidebar:
        st.header("Attention Settings")
        max_per_row = st.slider("Amino acids per row", 40, 100, 60, key="attention_max_per_row")
        top_k = st.slider("Top-k important regions", 3, 20, 10, key="attention_top_k")

    # --- Ligand and Protein Structure ---
    st.subheader("Ligand and Protein Structure")
    
    bottom = plt.get_cmap('Blues_r', 256)
    top = plt.get_cmap('Oranges', 256)
    newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), 
                            top(np.linspace(0.15, 0.65, 128))])
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    
    atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) for idx in range(len(atom_att))])
    radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
    ligand_img = clourMol(mol, highlightAtoms_p=range(len(atom_att)), 
                            highlightAtomColors_p=atom_color, radii=radii)

    col1, col2 = st.columns(2)
    with col1:
        st.image(ligand_img, caption="Molecule with Atom Importance as given by Grad-AAM from reference study (Yang et al. 2022).",
                 width=400)
    with col2:
        if 'uniprot_id' in output_dict:
            uniprot_id = output_dict['uniprot_id']
            st.markdown(f"**Protein 3D Structure for UniProt ID:** `{uniprot_id}`")
            html_3d = uniprot_to_html(uniprot_id)
            st.components.v1.html(html_3d, height=500, scrolling=True)

    # --- Attention Visualization ---
    st.subheader("Protein Sequence Attention")
    
    visualization_dict = create_attention_visualization(
        per_layer_cams, layer_names, protein_sequence[0], max_per_row
    )
    seq_letters = visualization_dict['seq_letters']
    fig_multi_layer_attention = visualization_dict['fig_multi_layer_attention']
    per_layer_cam_1d = visualization_dict['per_layer_cam_1d']

    # --- Mean Attention ---
    st.markdown("#### Mean Attention Across Layers")
    meaned_cam = np.mean(per_layer_cam_1d, axis=0)
    top_regions_msg = analyze_top_regions(meaned_cam, seq_letters, top_k=top_k)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.text_area(f"Top {top_k} Important Regions", value=top_regions_msg, height=300)
    with col2:
        st.markdown('Mean attention heatmap across all layers (colors are attention weight MinMax normalized).')
        fig_mean_attention = plot_sequence_attention(seq_letters, meaned_cam, max_per_row, title="Mean Protein Sequence Attention")
        st.pyplot(fig_mean_attention)
        if st.button("Save Attention Heatmap as SVG"):
            output_dir = "results/attention_analysis"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, "mean_attention_heatmap.svg")
            try:
                fig_mean_attention.savefig(filepath, format='svg', bbox_inches='tight')
                st.success(f"Saved to {filepath}")
            except Exception as e:
                st.error(f"Failed to save SVG. Error: {e}")


    # --- Mean Attention (Enhanced Contrast) ---
    with st.expander("Mean Attention (Enhanced Contrast)"):
        st.info("This plot uses a square root scale on the attention weights to make lower-attention regions more visible.")
        fig_enhanced_attention = plot_sequence_attention_enhanced(seq_letters, meaned_cam, max_per_row, title="Mean Protein Sequence Attention (Enhanced)")
        st.pyplot(fig_enhanced_attention)

    # --- Fraction of Importance ---
    st.subheader("Fraction of Importance")
    st.markdown("Quantifies how much of the protein sequence is needed to cover a certain percentage of the total attention mass.")
    
    protein_len = len(protein_input)
    cams_for_robustness = np.array(per_layer_cams)
    if cams_for_robustness.shape[1] > protein_len:
        cams_for_robustness = cams_for_robustness[:, :protein_len]

    st.markdown("**Fraction of Protein Sequence Needed for 80% Importance (Per Layer)**")
    _fractions_needed, _, _ = fraction_of_mass_per_layer(
        per_layer_cams=cams_for_robustness,
        target_fraction=0.8,
        protein_sequence=protein_input
    )
    # Replace key 'Layer {i}' with actual layer names:
    fractions_needed = {}
    for _lname, value in _fractions_needed.items():
        if _lname == 'mean':
            layer_name = 'Mean Across Layers'
        else:
            layer_idx = int(_lname.split('_')[1])
            layer_name = layer_names[layer_idx]
        fractions_needed[layer_name] = value
    df = pd.DataFrame.from_dict(fractions_needed, orient='index', columns=['Fraction of Protein Sequence'])
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.dataframe(df.style.format("{:.2%}"))

    # --- Multi-Layer Attention ---
    st.subheader("Multi-Layer Attention Visualization")
    st.plotly_chart(fig_multi_layer_attention, width='stretch')


    st.subheader("Naming Convention")
    st.markdown("""
    - **Layer 1**: The (only) CNN layer of the first StackCNN path.
    - **Layer 2.1**: The first CNN layer of the second StackCNN path.
    - **Layer 2.2**: The second CNN layer of the second StackCNN path.
    - **Layer 3.1**: The first CNN layer of the third StackCNN path.
    - **Layer 3.2**: The second CNN layer of the third StackCNN path.
    - **Layer 3.3**: The third CNN layer of the third StackCNN path.
    """)


def render_robustness_analysis(explainer, smiles_input, protein_input, sample_idx):
    st.header("🛡️ Robustness Analysis")
    st.markdown("Evaluates the model's sensitivity to masking important amino acids in the protein sequence.")
    
    data = ensure_main_explanation(explainer, smiles_input, protein_input, sample_idx)
    if not data: return

    output_dict = data['output']
    prediction = output_dict['prediction']
    per_layer_cams = output_dict['per_layer_cams']
    
    with st.sidebar:
        st.header("Robustness Settings")
        # Removed max_per_row as it was only used by sliding window, which is now gone.

    # --- Masking Robustness ---
    st.subheader("Masking Robustness")
    st.markdown("Measures how much the prediction changes when top-k important amino acids are masked, compared to random masking.")
    
    protein_len = len(protein_input)
    cams_for_robustness = np.array(per_layer_cams)
    if cams_for_robustness.shape[1] > protein_len:
        cams_for_robustness = cams_for_robustness[:, :protein_len]

    try:
        if os.path.exists(ROBUSTNESS_CACHE_FILE):
             df_mask = torch.load(ROBUSTNESS_CACHE_FILE, weights_only=False)
        else:
            with st.spinner("Calculating robustness to masking... This may take some time."):
                df_mask = test_robustness_masking(
                    explainer=explainer,
                    smiles_input=smiles_input,
                    protein_input=protein_input,
                    per_layer_cams=cams_for_robustness,
                    original_pred=prediction,
                    top_k_candidates=200,
                    top_k_masked=120,
                    n_repeats=10,
                )
                torch.save(df_mask, ROBUSTNESS_CACHE_FILE)
        
        fig_robustness = plot_robustness_bar(df_mask)
        st.pyplot(fig_robustness)

        st.markdown("**P-values for Top-k vs Random Δ Prediction:**")
        for i, row in df_mask.iterrows():
            st.markdown(f"- Token `{row['Mask Token']}`: p-value = `{row['p-value']:.4e}`")

    except Exception as e:
        st.error(f"Error during robustness masking: {str(e)}")

def render_gnn_explanations(explainer, gnn_explainer, smiles_input, protein_input, sample_idx):
    st.header("🧠 GNN Explanations")
    
    data = ensure_main_explanation(explainer, smiles_input, protein_input, sample_idx)
    if not data: return
    
    output_dict = data['output']

    # --- Ligand Image Generation/Caching ---
    if os.path.exists(LIGAND_IMG_CACHE_FILE):
        ligand_img = torch.load(LIGAND_IMG_CACHE_FILE, weights_only=False)
    else:
        atom_att = output_dict['atom_att']
        mol = output_dict['mol_obj']
        bottom = plt.get_cmap('Blues_r', 256)
        top = plt.get_cmap('Oranges', 256)
        newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), 
                                top(np.linspace(0.15, 0.65, 128))])
        newcmp = ListedColormap(newcolors, name='OrangeBlue')
        
        atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) 
                            for idx in range(len(atom_att))])
        radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
        ligand_img = clourMol(mol, highlightAtoms_p=range(len(atom_att)), 
                                highlightAtomColors_p=atom_color, radii=radii)
        torch.save(ligand_img, LIGAND_IMG_CACHE_FILE)

    # --- GNN Explanation ---
    with st.sidebar:
        st.header("GNN Settings")
        threshold = st.slider(
            "Edge Importance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.05,
            help="Adjust to show more or fewer edges based on their importance score.",
            key="gnn_edge_slider"
        )

    if os.path.exists(GNN_DATA_CACHE_FILE) and os.path.exists(GNN_OUTPUT_CACHE_FILE):
        with st.spinner("Loading cached GNN explanation..."):
            explanation_data = torch.load(GNN_DATA_CACHE_FILE, weights_only=False)
            gnn_output = torch.load(GNN_OUTPUT_CACHE_FILE, weights_only=False)
    else:
        with st.spinner("Generating GNN explanation... This may take a moment."):
            explanation_data = gnn_explainer.explain_attributes(smiles_input, protein_input)
            gnn_output = gnn_explainer.explain_graph(smiles_input, protein_input, sample_idx)
            torch.save(explanation_data, GNN_DATA_CACHE_FILE)
            torch.save(gnn_output, GNN_OUTPUT_CACHE_FILE)

    fig_feat, fig_graph = get_gnn_explanation_figures(*explanation_data, threshold=threshold)
    render_gnn_explanation_streamlit(fig_feat, fig_graph)
    
    gnn_figures = gnn_explainer.get_visualization_figures(gnn_output)
    gnn_main_visualisation_st(gnn_figures, gnn_output, ligand_img)


def render_interaction_analysis(explainer, smiles_input, protein_input, sample_idx):
    st.header("🤝 Interaction Analysis")
    
    data = ensure_main_explanation(explainer, smiles_input, protein_input, sample_idx)
    if not data: return
    
    output_dict = data['output']
    render_interaction_explanations(explainer, output_dict, smiles_input, protein_input)

def render_shap_explanations(explainer, smiles_input, protein_input, sample_idx):
    st.header("📊 SHAP Explanations")
    
    # Ensure base explanation just in case needed, though SHAP has its own run
    ensure_main_explanation(explainer, smiles_input, protein_input, sample_idx)

    if os.path.exists(SHAP_CACHE_FILE):
        with st.spinner("Loading cached SHAP explanation..."):
            shap_results = torch.load(SHAP_CACHE_FILE, weights_only=False)
    else:
        with st.spinner("Generating SHAP explanation... This may take a moment."):
            shap_results = explainer.explain_shap(smiles_input, protein_input)
            torch.save(shap_results, SHAP_CACHE_FILE)
    
    st.subheader("SHAP Kernel")
    display_shap_summary_streamlit(shap_results)
    
    shap_figs = plot_shap_explanation_plots(shap_results)
    for fig_name, fig_obj in shap_figs.items():
        st.pyplot(fig_obj)

    fig_waterfall = plot_shap_waterfall(shap_results)
    st.pyplot(fig_waterfall)

def render_lime_explanations(explainer, smiles_input, protein_input, sample_idx):
    st.header("🍋 LIME Explanations")
    
    ensure_main_explanation(explainer, smiles_input, protein_input, sample_idx)

    if os.path.exists(LIME_CACHE_FILE):
        with st.spinner("Loading cached LIME explanation..."):
            lime_output = torch.load(LIME_CACHE_FILE, weights_only=False)
    else:
        with st.spinner("Generating LIME explanation... This may take a moment."):
            lime_output = explainer.explain_lime(smiles_input, protein_input)
            torch.save(lime_output, LIME_CACHE_FILE)
            
    explanation_df = lime_output['explanation_df']

    fig, ax = get_styled_figure_ax(figsize=(10, 15), aspect='none', grid=True)
    
    ax.barh(
        explanation_df['feature'],
        explanation_df['weight'],
        color=[
            DATASET_COLORS[0] if c == 'green' else DATASET_COLORS[-1]
            for c in explanation_df['color']
        ]
    )
    
    ax.set_title('LIME Explanation for Protein Sequence')
    ax.set_xlabel('Weight (Contribution to Prediction)')
    ax.set_ylabel('Feature')
    
    # Generate y-tick positions and labels logic (simplified for brevity but preserving logic)
    all_y_ticks = np.arange(len(explanation_df['feature']))
    raw_formatted_labels = []
    for feature_label in explanation_df['feature']:
        parts = feature_label.split('=')
        if len(parts) == 2:
            pos_part = parts[0].strip().replace('pos_', 'Pos ')
            aa_part = parts[1].strip()
            raw_formatted_labels.append(f"{pos_part} ({aa_part})")
        else:
            raw_formatted_labels.append(feature_label)

    display_y_labels = [''] * len(raw_formatted_labels)
    indices_to_show = set(np.arange(0, len(raw_formatted_labels), 100))
    if len(raw_formatted_labels) > 0 and (len(raw_formatted_labels) - 1) not in indices_to_show:
        indices_to_show.add(len(raw_formatted_labels) - 1)
    
    for i in sorted(list(indices_to_show)):
        if i < len(raw_formatted_labels):
            display_y_labels[i] = raw_formatted_labels[i]

    ax.set_yticks(all_y_ticks)
    ax.set_yticklabels(display_y_labels)
    if len(indices_to_show) > 0:
        ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    
    st.pyplot(fig)

    # --- Top-k LIME Explanation ---
    st.subheader("Top Influential Positions")
    TOP_K_LIME = 20 

    top_k_df = explanation_df.copy()
    top_k_df['abs_weight'] = top_k_df['weight'].abs()
    top_k_df = top_k_df.sort_values(by='abs_weight', ascending=False).head(TOP_K_LIME)
    top_k_df = top_k_df.sort_values(by='weight', ascending=False)

    fig_top_k, ax_top_k = get_styled_figure_ax(figsize=(10, TOP_K_LIME * 0.7), aspect='none', grid=True)
    
    ax_top_k.barh(
        top_k_df['feature'],
        top_k_df['weight'],
        color=[
            DATASET_COLORS[0] if c == 'green' else DATASET_COLORS[-1] 
            for c in top_k_df['color']
        ]
    )
    
    ax_top_k.set_title(f'Top {TOP_K_LIME} LIME Explanations')
    ax_top_k.set_xlabel('Weight (Contribution to Prediction)')
    ax_top_k.set_ylabel('Feature')
    
    formatted_labels_top_k = []
    for feature_label in top_k_df['feature']:
        parts = feature_label.split('=')
        if len(parts) == 2:
            pos_part = parts[0].strip().replace('pos_', 'Pos ')
            aa_part = parts[1].strip()
            formatted_labels_top_k.append(f"{pos_part} ({aa_part})")
        else:
            formatted_labels_top_k.append(feature_label)

    ax_top_k.set_yticks(np.arange(len(formatted_labels_top_k)))
    ax_top_k.set_yticklabels(formatted_labels_top_k)
    ax_top_k.invert_yaxis() 
    
    st.pyplot(fig_top_k)
