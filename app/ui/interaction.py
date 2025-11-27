import numpy as np
import streamlit as st
import plotly.graph_objects as go
import os
import torch
import matplotlib.pyplot as plt
from src.utils.plot_utils import get_styled_figure_ax


from xai.interaction_explainer import (
    InteractionExplainer,
    visualize_interaction_matrix,
    visualize_marginal_attributions
)

from scipy.stats import spearmanr

# Define paths for cache files
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
GRADIENT_CACHE_FILE = os.path.join(CACHE_DIR, "interaction_gradient.pt")
IG_CACHE_FILE = os.path.join(CACHE_DIR, "interaction_ig.pt")
FEATURE_CACHE_FILE = os.path.join(CACHE_DIR, "interaction_feature.pt")





def render_interaction_explanations(explainer_instance, output_dict, smiles_input, protein_input):
    """
    Add interaction-level explanations section to the Streamlit app.
    """
    
    # Create interaction explainer
    interaction_explainer = InteractionExplainer(
        explainer_instance.model, 
        device=explainer_instance.device
    )
    
    # Get data
    data = output_dict.get('data_obj')
    if data is None:
        # Need to transform the input again
        data, gm = explainer_instance.dataset.transform_unique(smiles_input, protein_input)
        g, mol = gm
        data = data.to(explainer_instance.device)
    else:
        mol = output_dict['mol_obj']
    
    protein_sequence = output_dict['protein_sequence']
    
    # Define INT2VOCAB for amino acid labels
    from MGraphDTA.visualization.preprocessing import VOCAB_PROTEIN
    INT2VOCAB = {str(v): k for k, v in VOCAB_PROTEIN.items()}
    INT2VOCAB['0'] = '-'
    
    with st.sidebar:
        st.header("Interaction Settings")
        k_top_interactions = st.slider(
            "Top-k Interactions for Overlap",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Select the number of top interactions to compare for the overlap metric."
        )

    # ==============================
    # Section: Atom-to-Residue Interactions
    # ==============================
    st.header("🔗 Atom-to-Residue Interaction Analysis")
    
    st.markdown("""
    This section shows **which drug atoms interact with which protein residues** to produce 
    the binding affinity prediction. We use gradient-based attribution methods to identify 
    the most important atom-residue pairs.
    """)
    
    # Tab selection for different methods
    tab1, tab2, tab3 = st.tabs([
        "📊 Gradient Method (Fast)", 
        "🎯 Integrated Gradients (Accurate)",
        "📈 Feature-Level Interaction"
    ])
    
    # -----------------------------
    # TAB 1: Gradient Method
    # -----------------------------
    with tab1:
        st.markdown("### Simple Gradient-Based Interaction Map")
        st.info("Computes gradients of the prediction w.r.t. atom and residue features, "
                "then creates an interaction matrix via outer product.")
        
        if os.path.exists(GRADIENT_CACHE_FILE):
            with st.spinner("Loading cached gradient-based interactions..."):
                gradient_result = torch.load(GRADIENT_CACHE_FILE, weights_only=False)
        else:
            with st.spinner("Computing gradient-based interactions..."):
                gradient_result = interaction_explainer.gradient_atom_residue_map(
                    data, 
                    protein_sequence
                )
                torch.save(gradient_result, GRADIENT_CACHE_FILE)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Atoms", gradient_result['num_atoms'])
        with col2:
            st.metric("Number of Residues", gradient_result['num_residues'])
        with col3:
            n_interactions = gradient_result['num_atoms'] * gradient_result['num_residues']
            st.metric("Total Interactions", f"{n_interactions:,}")
        
        st.subheader("Interaction Heatmap (Gradient Method)")
        fig_heatmap = visualize_interaction_matrix(
            gradient_result,
            protein_sequence,
            mol,
            int2vocab=INT2VOCAB,
            title="Atom-to-Residue Interaction Heatmap (Gradient)",
            max_residues=1200,
            figsize=(15, 25)
        )
        st.pyplot(fig_heatmap)
        if st.button("Save Gradient Heatmap as SVG"):
            output_dir = "results/interaction_analysis"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, "interaction_heatmap_gradient.svg")
            try:
                fig_heatmap.savefig(filepath, format='svg', bbox_inches='tight')
                st.success(f"Saved to {filepath}")
            except Exception as e:
                st.error(f"Failed to save SVG. Error: {e}")
        
        # Visualization: Marginal Attributions
        st.subheader("Marginal Attribution Scores (Gradient Method)")
        st.markdown("Shows the most important atoms and residues by summing interactions across the other dimension.")
        fig_marginal = visualize_marginal_attributions(
            gradient_result,
            protein_sequence,
            mol,
            int2vocab=INT2VOCAB,
            max_display=30
        )
        st.plotly_chart(fig_marginal, use_container_width=True)
        # add_save_buttons(fig_marginal, "marginal_attribution_gradient")
        
        # Top interactions table
        st.subheader("Top 20 Atom-Residue Interactions (Gradient Method)")
        top_interactions_df = interaction_explainer.get_top_interactions(
            gradient_result['interaction_matrix'],
            protein_sequence,
            mol,
            top_k=20,
            int2vocab=INT2VOCAB
        )
        st.dataframe(
            top_interactions_df.style.background_gradient(
                subset=['interaction_score'], 
                cmap='RdBu', 
                vmin=-top_interactions_df['interaction_score'].abs().max(),
                vmax=top_interactions_df['interaction_score'].abs().max()
            ),
            use_container_width=True
        )
        
        # Download button
        csv = top_interactions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Top Interactions (CSV)",
            data=csv,
            file_name="atom_residue_interactions_gradient.csv",
            mime="text/csv"
        )
    
    # -----------------------------
    # TAB 2: Integrated Gradients
    # -----------------------------
    with tab2:
        st.markdown("### Integrated Gradients Interaction Map")
        st.info("Uses Integrated Gradients for more stable and theoretically grounded attributions. "
                "This method accumulates gradients along a path from baseline to input.")
        
        # Settings for IG
        n_steps = 150
        
        if os.path.exists(IG_CACHE_FILE):
            with st.spinner("Loading cached Integrated Gradients..."):
                ig_result = torch.load(IG_CACHE_FILE, weights_only=False)
        else:
            with st.spinner(f"Computing Integrated Gradients with {n_steps} steps... This is slow."):
                ig_result = interaction_explainer.integrated_gradients_interaction(
                    data,
                    protein_sequence,
                    n_steps=n_steps
                )
                torch.save(ig_result, IG_CACHE_FILE)
        
        st.success("✅ Integrated Gradients computed successfully!")
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Atoms", ig_result['num_atoms'])
        with col2:
            st.metric("Number of Residues", ig_result['num_residues'])
        with col3:
            st.metric("Method", "Integrated Gradients")
        
        st.subheader("Interaction Heatmap (Integrated Gradients)")
        fig_heatmap_ig = visualize_interaction_matrix(
            ig_result,
            protein_sequence,
            mol,
            int2vocab=INT2VOCAB,
            title="Atom-to-Residue Interaction Heatmap (Integrated Gradients)",
            max_residues=1200,
            figsize=(15, 25)
        )
        st.pyplot(fig_heatmap_ig)
        if st.button("Save IG Heatmap as SVG"):
            output_dir = "results/interaction_analysis"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, "interaction_heatmap_ig.svg")
            try:
                fig_heatmap_ig.savefig(filepath, format='svg', bbox_inches='tight')
                st.success(f"Saved to {filepath}")
            except Exception as e:
                st.error(f"Failed to save SVG. Error: {e}")

        # Visualization: Marginal Attributions
        st.subheader("Marginal Attribution Scores (Integrated Gradients)")
        fig_marginal_ig = visualize_marginal_attributions(
            ig_result,
            protein_sequence,
            mol,
            int2vocab=INT2VOCAB,
            max_display=30
        )
        st.plotly_chart(fig_marginal_ig, use_container_width=True)
        # add_save_buttons(fig_marginal_ig, "marginal_attribution_ig")
        
        # Top interactions table
        st.subheader("Top 20 Atom-Residue Interactions (Integrated Gradients)")
        top_interactions_ig_df = interaction_explainer.get_top_interactions(
            ig_result['interaction_matrix'],
            protein_sequence,
            mol,
            top_k=20,
            int2vocab=INT2VOCAB
        )
        st.dataframe(
            top_interactions_ig_df.style.background_gradient(
                subset=['interaction_score'], 
                cmap='RdBu',
                vmin=-top_interactions_ig_df['interaction_score'].abs().max(),
                vmax=top_interactions_ig_df['interaction_score'].abs().max()
            ),
            use_container_width=True
        )
        
        # Download button
        csv_ig = top_interactions_ig_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Top Interactions (CSV)",
            data=csv_ig,
            file_name="atom_residue_interactions_ig.csv",
            mime="text/csv"
        )

    # -----------------------------
    # TAB 3: Feature-Level Interaction
    # -----------------------------
    with tab3:
        st.markdown("### Feature-Level Interaction Map")
        st.info("Shows interactions at the learned feature level (after encoding). "
                "This gives a 96×96 matrix showing how protein and ligand features interact.")
        
        if os.path.exists(FEATURE_CACHE_FILE):
            with st.spinner("Loading cached feature-level interactions..."):
                feature_result = torch.load(FEATURE_CACHE_FILE, weights_only=False)
        else:
            with st.spinner("Computing feature-level interactions..."):
                feature_result = interaction_explainer.gradient_interaction_map(data)
                torch.save(feature_result, FEATURE_CACHE_FILE)

        # Display statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Protein Features", len(feature_result['protein_attributions']))
        with col2:
            st.metric("Ligand Features", len(feature_result['ligand_attributions']))
        
        # Visualization: Feature interaction heatmap
        st.subheader("Feature-Level Interaction Matrix")
        
        fig_feature, ax = get_styled_figure_ax(figsize=(10, 8), aspect='equal', grid=False)
        interaction_map = feature_result['interaction_map']
        abs_max = np.abs(interaction_map).max()
        im = ax.imshow(interaction_map, cmap='RdBu', vmin=-abs_max, vmax=abs_max)
        fig_feature.colorbar(im, ax=ax, label="Interaction Score")
        ax.set_title("Protein-Ligand Feature Interaction Matrix")
        ax.set_xlabel("Ligand Features")
        ax.set_ylabel("Protein Features")
        ax.set_xticks(np.arange(0, 96, 10))
        ax.set_yticks(np.arange(0, 96, 10))
        
        st.pyplot(fig_feature)
        if st.button("Save Feature Heatmap as SVG"):
            output_dir = "results/interaction_analysis"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, "feature_interaction_matrix.svg")
            try:
                fig_feature.savefig(filepath, format='svg', bbox_inches='tight')
                st.success(f"Saved to {filepath}")
            except Exception as e:
                st.error(f"Failed to save SVG. Error: {e}")

        # Feature attribution bar charts
        st.subheader("Feature Attribution Scores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Protein features
            fig_protein = go.Figure(go.Bar(
                x=list(range(len(feature_result['protein_attributions']))),
                y=feature_result['protein_attributions'],
                marker_color=['red' if x < 0 else 'blue' 
                             for x in feature_result['protein_attributions']]
            ))
            fig_protein.update_layout(
                title="Protein Feature Attributions",
                xaxis_title="Feature Index",
                yaxis_title="Attribution Score",
                height=400
            )
            st.plotly_chart(fig_protein, use_container_width=True)
            # add_save_buttons(fig_protein, "protein_feature_attributions")
        
        with col2:
            # Ligand features
            fig_ligand = go.Figure(go.Bar(
                x=list(range(len(feature_result['ligand_attributions']))),
                y=feature_result['ligand_attributions'],
                marker_color=['red' if x < 0 else 'blue' 
                             for x in feature_result['ligand_attributions']]
            ))
            fig_ligand.update_layout(
                title="Ligand Feature Attributions",
                xaxis_title="Feature Index",
                yaxis_title="Attribution Score",
                height=400
            )
            st.plotly_chart(fig_ligand, use_container_width=True)
            # add_save_buttons(fig_ligand, "ligand_feature_attributions")
    
    # ==============================
    # Section: Method Comparison
    # ==============================
    # Only show comparison if both results are available (computed)
    if os.path.exists(GRADIENT_CACHE_FILE) and os.path.exists(IG_CACHE_FILE):
        st.header("⚖️ Method Comparison")
        st.markdown("Comparing gradient-based vs. integrated gradients methods")
        
        gradient_result = torch.load(GRADIENT_CACHE_FILE, weights_only=False)
        ig_result = torch.load(IG_CACHE_FILE, weights_only=False)
        
        # Compare top interactions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 Interactions (Gradient)")
            top_grad = interaction_explainer.get_top_interactions(
                gradient_result['interaction_matrix'],
                protein_sequence,
                mol,
                top_k=10,
                int2vocab=INT2VOCAB
            )
            st.dataframe(top_grad[['residue_idx', 'atom_idx', 'residue_aa', 
                                   'atom_symbol', 'interaction_score']])
        
        with col2:
            st.markdown("#### Top 10 Interactions (Integrated Gradients)")
            top_ig = interaction_explainer.get_top_interactions(
                ig_result['interaction_matrix'],
                protein_sequence,
                mol,
                top_k=10,
                int2vocab=INT2VOCAB
            )
            st.dataframe(top_ig[['residue_idx', 'atom_idx', 'residue_aa', 
                                 'atom_symbol', 'interaction_score']])
        
        # Similarity Analysis
        st.markdown("#### Similarity Analysis of Interaction Matrix")
        st.markdown("A deeper look at how the two methods compare on the full interaction matrix.")

        # Prepare flattened data
        grad_matrix = gradient_result['interaction_matrix']
        ig_matrix = ig_result['interaction_matrix']
        non_padding_mask = gradient_result.get('non_padding_mask')

        if non_padding_mask is not None and grad_matrix.shape[0] == len(non_padding_mask) and ig_matrix.shape[0] == len(non_padding_mask):
            grad_flat = grad_matrix[non_padding_mask, :].flatten()
            ig_flat = ig_matrix[non_padding_mask, :].flatten()
        else:
            st.warning("Matrix shapes do not match padding mask. Using full matrices for correlation.")
            grad_flat = grad_matrix.flatten()
            ig_flat = ig_matrix.flatten()

        min_len = min(len(grad_flat), len(ig_flat))
        grad_flat = grad_flat[:min_len]
        ig_flat = ig_flat[:min_len]

        # --- Softer & Harder Metrics ---
        col1, col2, col3 = st.columns(3)
        
        # Pearson Correlation (Standard)
        with col1:
            st.markdown("**Standard Correlation**")
            pearson_corr = np.corrcoef(grad_flat, ig_flat)[0, 1]
            st.metric("Pearson Correlation", f"{pearson_corr:.4f}", help="Measures linear relationship. Sensitive to outliers.")

        # Spearman Correlation (Softer)
        with col2:
            st.markdown("**Rank-based Correlation (Softer)**")
            spearman_corr, _ = spearmanr(grad_flat, ig_flat)
            st.metric("Spearman Correlation", f"{spearman_corr:.4f}", help="Measures monotonic relationship (rank similarity). Less sensitive to outliers.")

        # Mean Absolute Error (Harder)
        with col3:
            st.markdown("**Error-based Metric (Harder)**")
            mae = np.mean(np.abs(grad_flat - ig_flat))
            st.metric("Mean Absolute Error", f"{mae:.4f}", help="Average absolute difference between scores. Lower is better.")

        # --- Top-k Intersection (Harder) ---
        st.markdown("**Top-k Interaction Overlap (Harder)**")
        
        # Get top-k indices for each method
        top_k_grad_indices = np.argsort(np.abs(grad_flat))[-k_top_interactions:]
        top_k_ig_indices = np.argsort(np.abs(ig_flat))[-k_top_interactions:]
        
        # Calculate intersection
        intersection = np.intersect1d(top_k_grad_indices, top_k_ig_indices)
        overlap_percentage = (len(intersection) / k_top_interactions) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Overlap in Top {k_top_interactions} Interactions", f"{overlap_percentage:.2f}%")
        with col2:
            st.progress(overlap_percentage / 100)

        # --- Scatter Plot ---
        with st.expander("Visual Correlation Scatter Plot for Interaction Matrix"):
            # Subsample for performance
            subsample_indices = np.random.choice(len(grad_flat), size=min(len(grad_flat), 5000), replace=False)
            
            fig_corr = go.Figure(go.Scatter(
                x=grad_flat[subsample_indices],
                y=ig_flat[subsample_indices],
                mode='markers',
                marker=dict(size=4, opacity=0.6, color=pearson_corr, colorscale='Viridis', cmin=-1, cmax=1),
                name='Interactions'
            ))
            fig_corr.update_layout(
                title="Method Correlation (Subsampled)",
                xaxis_title="Gradient Method Score",
                yaxis_title="Integrated Gradients Score",
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            # add_save_buttons(fig_corr, "correlation_scatter_plot")

        # ==========================================================
        # SECTION: Comparison of Marginal Attributions
        # ==========================================================
        st.markdown("---")
        st.markdown("#### Similarity Analysis of Marginal Attributions")
        st.markdown("Comparing the importance scores assigned to individual atoms and residues.")

        # --- Prepare Atom and Residue Attributions ---
        grad_atom_attr = gradient_result['atom_attributions']
        ig_atom_attr = ig_result['atom_attributions']
        
        grad_residue_attr = gradient_result['residue_attributions']
        ig_residue_attr = ig_result['residue_attributions']

        non_padding_mask = gradient_result.get('non_padding_mask')

        if non_padding_mask is not None and len(grad_residue_attr) == len(non_padding_mask) and len(ig_residue_attr) == len(non_padding_mask):
            grad_residue_attr = grad_residue_attr[non_padding_mask]
            ig_residue_attr = ig_residue_attr[non_padding_mask]

        min_len_atoms = min(len(grad_atom_attr), len(ig_atom_attr))
        grad_atom_attr = grad_atom_attr[:min_len_atoms]
        ig_atom_attr = ig_atom_attr[:min_len_atoms]

        min_len_residues = min(len(grad_residue_attr), len(ig_residue_attr))
        grad_residue_attr = grad_residue_attr[:min_len_residues]
        ig_residue_attr = ig_residue_attr[:min_len_residues]

        # Use 30% of the number of atoms/residues for top-k
        k_top_attributions_atoms = max(1, int(0.3 * len(grad_atom_attr)))
        k_top_attributions_residues = max(1, int(0.3 * len(grad_residue_attr)))

        attr_tab1, attr_tab2 = st.tabs(["Atom Attributions Comparison", "Residue Attributions Comparison"])

        with attr_tab1:
            st.markdown("##### Atom Attribution Similarity")
            # --- Metrics ---
            col1, col2, col3 = st.columns(3)
            with col1:
                pearson_corr_atom = np.corrcoef(grad_atom_attr, ig_atom_attr)[0, 1]
                st.metric("Pearson Correlation", f"{pearson_corr_atom:.4f}")
            with col2:
                spearman_corr_atom, _ = spearmanr(grad_atom_attr, ig_atom_attr)
                st.metric("Spearman Correlation", f"{spearman_corr_atom:.4f}")
            with col3:
                mae_atom = np.mean(np.abs(grad_atom_attr - ig_atom_attr))
                st.metric("Mean Absolute Error", f"{mae_atom:.4f}")

            # --- Top-k Overlap ---
            st.markdown("**Top-k Atom Overlap**")
            top_k_grad_indices_atom = np.argsort(np.abs(grad_atom_attr))[-k_top_attributions_atoms:]
            top_k_ig_indices_atom = np.argsort(np.abs(ig_atom_attr))[-k_top_attributions_atoms:]
            
            intersection_atom = np.intersect1d(top_k_grad_indices_atom, top_k_ig_indices_atom)
            overlap_percentage_atom = (len(intersection_atom) / k_top_attributions_atoms) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Overlap in Top {k_top_attributions_atoms} Atoms", f"{overlap_percentage_atom:.2f}%")
            with col2:
                st.progress(overlap_percentage_atom / 100)

            # --- Scatter Plot ---
            with st.expander("Visual Correlation Scatter Plot for Atoms"):
                fig_corr_atom = go.Figure(go.Scatter(
                    x=grad_atom_attr,
                    y=ig_atom_attr,
                    mode='markers',
                    marker=dict(size=5, opacity=0.7),
                    name='Atoms'
                ))
                fig_corr_atom.update_layout(
                    title="Atom Attribution Correlation",
                    xaxis_title="Gradient Method Atom Score",
                    yaxis_title="Integrated Gradients Atom Score",
                    height=500
                )
                st.plotly_chart(fig_corr_atom, use_container_width=True)
                # add_save_buttons(fig_corr_atom, "atom_attribution_correlation")

        with attr_tab2:
            st.markdown("##### Residue Attribution Similarity")
            # --- Metrics ---
            col1, col2, col3 = st.columns(3)
            with col1:
                pearson_corr_res = np.corrcoef(grad_residue_attr, ig_residue_attr)[0, 1]
                st.metric("Pearson Correlation", f"{pearson_corr_res:.4f}")
            with col2:
                spearman_corr_res, _ = spearmanr(grad_residue_attr, ig_residue_attr)
                st.metric("Spearman Correlation", f"{spearman_corr_res:.4f}")
            with col3:
                mae_res = np.mean(np.abs(grad_residue_attr - ig_residue_attr))
                st.metric("Mean Absolute Error", f"{mae_res:.4f}")

            # --- Top-k Overlap ---
            st.markdown("**Top-k Residue Overlap**")
            top_k_grad_indices_res = np.argsort(np.abs(grad_residue_attr))[-k_top_attributions_residues:]
            top_k_ig_indices_res = np.argsort(np.abs(ig_residue_attr))[-k_top_attributions_residues:]
            
            intersection_res = np.intersect1d(top_k_grad_indices_res, top_k_ig_indices_res)
            overlap_percentage_res = (len(intersection_res) / k_top_attributions_residues) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Overlap in Top {k_top_attributions_residues} Residues", f"{overlap_percentage_res:.2f}%")
            with col2:
                st.progress(overlap_percentage_res / 100)
            
            # --- Scatter Plot ---
            with st.expander("Visual Correlation Scatter Plot for Residues"):
                fig_corr_res = go.Figure(go.Scatter(
                    x=grad_residue_attr,
                    y=ig_residue_attr,
                    mode='markers',
                    marker=dict(size=5, opacity=0.7),
                    name='Residues'
                ))
                fig_corr_res.update_layout(
                    title="Residue Attribution Correlation",
                    xaxis_title="Gradient Method Residue Score",
                    yaxis_title="Integrated Gradients Residue Score",
                    height=500
                )
                st.plotly_chart(fig_corr_res, use_container_width=True)
                # add_save_buttons(fig_corr_res, "residue_attribution_correlation")

