import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import os
import json
import glob

def calculate_centroid(matrix):
    """Calculate the center of mass (centroid) of a 2D matrix."""
    # Normalize the matrix to be a probability distribution
    matrix = np.abs(matrix)
    if np.sum(matrix) == 0:
        return np.array([0, 0])
    matrix = matrix / np.sum(matrix)
    
    # Get coordinates and weights
    y_coords, x_coords = np.indices(matrix.shape)
    
    # Calculate centroid
    centroid_y = np.sum(y_coords * matrix)
    centroid_x = np.sum(x_coords * matrix)
    
    return np.array([centroid_y, centroid_x])

def load_analysis_results(results_dir):
    """Loads all analysis results from the specified directory."""
    if not os.path.isdir(results_dir):
        st.error(f"Results directory not found: {results_dir}")
        return None

    try:
        # Load metadata
        with open(os.path.join(results_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Load results dataframe
        results_df = pd.read_csv(os.path.join(results_dir, "results.csv"))

        def load_batched_npz_archives(base_filepath_pattern):
            all_arrays = []
            # Find all batch files, sort them numerically by batch number
            batch_files = sorted(
                glob.glob(base_filepath_pattern), 
                key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
            )
            for filepath in batch_files:
                with np.load(filepath) as data:
                    # data.files gives keys like 'arr_0', 'arr_1', ...
                    all_arrays.extend([data[key] for key in sorted(data.files, key=lambda k: int(k.split('_')[1]))])
            return all_arrays

        all_grad_atom_residue_matrices = load_batched_npz_archives(os.path.join(results_dir, "grad_atom_residue_matrices_batch_*.npz"))
        all_ig_atom_residue_matrices = load_batched_npz_archives(os.path.join(results_dir, "ig_atom_residue_matrices_batch_*.npz"))
        all_protein_embeddings = load_batched_npz_archives(os.path.join(results_dir, "protein_embeddings_batch_*.npz"))
        all_ligand_embeddings = load_batched_npz_archives(os.path.join(results_dir, "ligand_embeddings_batch_*.npz"))
        all_protein_attributions = load_batched_npz_archives(os.path.join(results_dir, "protein_attributions_batch_*.npz"))
        all_ligand_attributions = load_batched_npz_archives(os.path.join(results_dir, "ligand_attributions_batch_*.npz"))

        num_residues_list = np.load(os.path.join(results_dir, "num_residues_list.npy")).tolist()
        num_atoms_list = np.load(os.path.join(results_dir, "num_atoms_list.npy")).tolist()

        with np.load(os.path.join(results_dir, "top_features.npz")) as data:
            all_top_protein_features = data['protein_features'].tolist()
            all_top_ligand_features = data['ligand_features'].tolist()
            all_top_interaction_scores = data['interaction_scores'].tolist()

        return {
            "metadata": metadata,
            "results_df": results_df,
            "all_grad_atom_residue_matrices": all_grad_atom_residue_matrices,
            "all_ig_atom_residue_matrices": all_ig_atom_residue_matrices,
            "num_residues_list": num_residues_list,
            "num_atoms_list": num_atoms_list,
            "all_top_protein_features": all_top_protein_features,
            "all_top_ligand_features": all_top_ligand_features,
            "all_top_interaction_scores": all_top_interaction_scores,
            "all_protein_embeddings": all_protein_embeddings,
            "all_ligand_embeddings": all_ligand_embeddings,
            "all_protein_attributions": all_protein_attributions,
            "all_ligand_attributions": all_ligand_attributions,
        }
    except FileNotFoundError as e:
        st.error(f"Missing file in results directory: {e.filename}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading results: {e}")
        return None

def main():
    st.set_page_config(page_title="Batch Interaction Analysis", layout="wide")
    st.title("🔬 Batch Analysis of Interaction Explainers")

    st.markdown("""
    This tool visualizes the results of a batch analysis of interaction explanation methods.
    Run the analysis separately using `run_batch_analysis.py` and then provide the results directory here.
    """)

    with st.sidebar:
        st.header("Settings")
        results_dir = st.text_input("Results Directory", "interaction_results/run_1")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Use this tool to get statistically robust comparisons of explanation methods.")

    if st.button("📊 Load and Visualize Results", type="primary"):
        with st.spinner("Loading analysis results..."):
            loaded_data = load_analysis_results(results_dir)

        if loaded_data:
            st.success(f"✅ Results loaded successfully from `{results_dir}`")

            metadata = loaded_data["metadata"]
            results_df = loaded_data["results_df"]
            all_grad_atom_residue_matrices = loaded_data["all_grad_atom_residue_matrices"]
            all_ig_atom_residue_matrices = loaded_data["all_ig_atom_residue_matrices"]
            num_residues_list = loaded_data["num_residues_list"]
            num_atoms_list = loaded_data["num_atoms_list"]
            all_top_protein_features = loaded_data["all_top_protein_features"]
            all_top_ligand_features = loaded_data["all_top_ligand_features"]
            all_top_interaction_scores = loaded_data["all_top_interaction_scores"]
            all_protein_embeddings = loaded_data["all_protein_embeddings"]
            all_ligand_embeddings = loaded_data["all_ligand_embeddings"]
            all_protein_attributions = loaded_data["all_protein_attributions"]
            all_ligand_attributions = loaded_data["all_ligand_attributions"]
            
            k_top_interactions = metadata.get("k_top_interactions", "N/A")
            num_samples = metadata.get("num_samples", "N/A")

            # --- Calculate Average Atom-Residue Interaction Matrices ---
            max_num_residues = max(num_residues_list) if num_residues_list else 0
            max_num_atoms = max(num_atoms_list) if num_atoms_list else 0

            avg_grad_ar_matrix = np.zeros((max_num_residues, max_num_atoms))
            avg_ig_ar_matrix = np.zeros((max_num_residues, max_num_atoms))
            ar_count_matrix = np.zeros((max_num_residues, max_num_atoms))

            for i in range(len(all_grad_atom_residue_matrices)):
                grad_mat = all_grad_atom_residue_matrices[i]
                ig_mat = all_ig_atom_residue_matrices[i]
                
                padded_grad_mat = np.pad(grad_mat, ((0, max_num_residues - grad_mat.shape[0]), (0, max_num_atoms - grad_mat.shape[1])), 'constant')
                padded_ig_mat = np.pad(ig_mat, ((0, max_num_residues - ig_mat.shape[0]), (0, max_num_atoms - ig_mat.shape[1])), 'constant')

                avg_grad_ar_matrix += padded_grad_mat
                avg_ig_ar_matrix += padded_ig_mat
                ar_count_matrix[0:grad_mat.shape[0], 0:grad_mat.shape[1]] += 1
            
            avg_grad_ar_matrix = np.divide(avg_grad_ar_matrix, ar_count_matrix, out=np.zeros_like(avg_grad_ar_matrix), where=ar_count_matrix!=0)
            avg_ig_ar_matrix = np.divide(avg_ig_ar_matrix, ar_count_matrix, out=np.zeros_like(avg_ig_ar_matrix), where=ar_count_matrix!=0)

            # --- Display Results ---
            st.header("↔️ Side-by-Side Average Atom-Residue Interactions")
            st.markdown("Compare the average interaction scores from Gradient and Integrated Gradients methods. "
                        "Scores are normalized per sample and then averaged.")

            global_max_abs_val = max(np.max(np.abs(avg_grad_ar_matrix)), np.max(np.abs(avg_ig_ar_matrix)))
            if global_max_abs_val == 0:
                global_max_abs_val = 1

            col_grad, col_ig = st.columns(2)

            st.header("Summary per residue/atom")
            avg_grad_residue_scores = np.sum(np.abs(avg_grad_ar_matrix), axis=1)
            avg_ig_residue_scores = np.sum(np.abs(avg_ig_ar_matrix), axis=1)
            fig_residue = go.Figure()
            fig_residue.add_trace(go.Bar(x=list(range(len(avg_grad_residue_scores))), y=avg_grad_residue_scores, name='Gradient Residue Scores', marker_color='blue', opacity=0.6))
            fig_residue.add_trace(go.Bar(x=list(range(len(avg_ig_residue_scores))), y=avg_ig_residue_scores, name='IG Residue Scores', marker_color='orange', opacity=0.6))
            fig_residue.update_layout(title="Average Residue Attribution Scores", xaxis_title="Residue Index", yaxis_title="Total Attribution Score", barmode='overlay', height=500)
            st.plotly_chart(fig_residue, use_container_width=True)

            st.header("📊 Aggregated Comparison Metrics")
            st.dataframe(results_df.describe().T)

            st.header("🔬 Embedding Interaction Analysis")
            st.markdown(f"Frequency of embedding dimensions appearing in the top {k_top_interactions} interactions across {num_samples} samples.")

            interaction_pairs = list(zip(all_top_protein_features, all_top_ligand_features))
            interaction_counts = pd.Series(interaction_pairs).value_counts()
            
            freq_matrix = np.zeros((96, 96))
            for (p_idx, l_idx), count in interaction_counts.items():
                freq_matrix[p_idx, l_idx] = count

            fig_heatmap = go.Figure(data=go.Heatmap(z=freq_matrix, colorscale='reds', colorbar=dict(title="Frequency")))
            fig_heatmap.update_layout(title="Protein-Ligand Feature Interaction Frequency Hotspots", xaxis_title="Ligand Feature Dimension", yaxis_title="Protein Feature Dimension", width=800, height=800)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("### Average Contribution to Prediction")
            st.markdown("This density map shows the average interaction score for each feature pair. Positive (blue) scores augment the prediction, while negative (red) scores diminish it.")

            sum_matrix = np.zeros((96, 96))
            count_matrix = np.zeros((96, 96))

            for i in range(len(all_top_protein_features)):
                p_idx = all_top_protein_features[i]
                l_idx = all_top_ligand_features[i]
                score = all_top_interaction_scores[i]
                sum_matrix[p_idx, l_idx] += score
                count_matrix[p_idx, l_idx] += 1

            avg_score_matrix = np.divide(sum_matrix, count_matrix, out=np.zeros_like(sum_matrix), where=count_matrix!=0)

            fig_contour = go.Figure(data=go.Contour(z=avg_score_matrix, colorscale='RdBu', zmid=0, colorbar=dict(title="Avg. Score")))
            fig_contour.update_layout(title="Average Interaction Score Landscape", xaxis_title="Ligand Feature Dimension", yaxis_title="Protein Feature Dimension", width=800, height=800)
            st.plotly_chart(fig_contour, use_container_width=True)

            st.header("🗺️ 2D Embedding Maps (t-SNE)")
            st.markdown("Visualize high-dimensional protein and ligand features reduced to 2D, colored by average attribution.")

            if len(all_protein_embeddings) > 0 and len(all_ligand_embeddings) > 0:
                protein_embeddings_agg = np.vstack(all_protein_embeddings)
                ligand_embeddings_agg = np.vstack(all_ligand_embeddings)
                protein_attributions_agg = np.concatenate(all_protein_attributions)
                ligand_attributions_agg = np.concatenate(all_ligand_attributions)

                if protein_embeddings_agg.shape[0] > 1 and protein_embeddings_agg.shape[1] > 1:
                    n_samples_protein = protein_embeddings_agg.shape[0]
                    perplexity_protein = min(30, n_samples_protein - 1)
                    reducer_protein = TSNE(n_components=2, random_state=42, perplexity=perplexity_protein)
                    embedding_protein_2d = reducer_protein.fit_transform(protein_embeddings_agg)

                    fig_protein_tsne = go.Figure(data=go.Scatter(
                        x=embedding_protein_2d[:, 0], y=embedding_protein_2d[:, 1], mode='markers',
                        marker=dict(size=5, color=protein_attributions_agg, colorscale='Viridis', colorbar=dict(title="Avg. Attribution"), showscale=True),
                        hoverinfo='text', hovertext=[f"Attribution: {a:.4f}" for a in protein_attributions_agg]
                    ))
                    fig_protein_tsne.update_layout(title="Protein Embedding t-SNE (Colored by Attribution)", xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2", width=800, height=600)
                    st.plotly_chart(fig_protein_tsne, use_container_width=True)
                else:
                    st.info("Not enough protein embeddings to perform t-SNE.")

                if ligand_embeddings_agg.shape[0] > 1 and ligand_embeddings_agg.shape[1] > 1:
                    n_samples_ligand = ligand_embeddings_agg.shape[0]
                    perplexity_ligand = min(30, n_samples_ligand - 1)
                    reducer_ligand = TSNE(n_components=2, random_state=42, perplexity=perplexity_ligand)
                    embedding_ligand_2d = reducer_ligand.fit_transform(ligand_embeddings_agg)

                    fig_ligand_tsne = go.Figure(data=go.Scatter(
                        x=embedding_ligand_2d[:, 0], y=embedding_ligand_2d[:, 1], mode='markers',
                        marker=dict(size=5, color=ligand_attributions_agg, colorscale='Plasma', colorbar=dict(title="Avg. Attribution"), showscale=True),
                        hoverinfo='text', hovertext=[f"Attribution: {a:.4f}" for a in ligand_attributions_agg]
                    ))
                    fig_ligand_tsne.update_layout(title="Ligand Embedding t-SNE (Colored by Attribution)", xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2", width=800, height=600)
                    st.plotly_chart(fig_ligand_tsne, use_container_width=True)
                else:
                    st.info("Not enough ligand embeddings to perform t-SNE.")
            else:
                st.info("No embedding data collected for t-SNE visualization. Run batch analysis first.")

if __name__ == "__main__":
    main()
