import os
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.manifold import TSNE

def load_analysis_results(results_dir):
    """Loads all analysis results from the specified directory."""
    if not os.path.isdir(results_dir):
        st.error(f"Results directory not found: {results_dir}")
        return None

    try:
        with open(os.path.join(results_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        results_df = pd.read_csv(os.path.join(results_dir, "results.csv"))

        def load_batched_npz(pattern):
            all_arrays = []
            files = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
            for f in files:
                with np.load(f) as data:
                    all_arrays.extend([data[key] for key in sorted(data.files, key=lambda k: int(k.split('_')[1]))])
            return all_arrays

        return {
            "metadata": metadata,
            "results_df": results_df,
            "all_grad_atom_residue_matrices": load_batched_npz(os.path.join(results_dir, "grad_matrices_batch_*.npz")),
            "all_ig_atom_residue_matrices": load_batched_npz(os.path.join(results_dir, "ig_matrices_batch_*.npz")),
            "all_protein_embeddings": load_batched_npz(os.path.join(results_dir, "protein_embeddings_batch_*.npz")),
            "all_ligand_embeddings": load_batched_npz(os.path.join(results_dir, "ligand_embeddings_batch_*.npz")),
            "all_protein_attributions": load_batched_npz(os.path.join(results_dir, "protein_attributions_batch_*.npz")),
            "all_ligand_attributions": load_batched_npz(os.path.join(results_dir, "ligand_attributions_batch_*.npz")),
            "num_residues_list": np.load(os.path.join(results_dir, "num_residues_list.npy")).tolist(),
            "num_atoms_list": np.load(os.path.join(results_dir, "num_atoms_list.npy")).tolist(),
            "top_features": np.load(os.path.join(results_dir, "top_features.npz"))
        }
    except FileNotFoundError as e:
        st.error(f"Missing file in results directory: {e.filename}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading results: {e}")
        return None

def process_analysis_results(loaded_data):
    """Processes the loaded data to generate visualizations and summaries."""
    
    # Average Atom-Residue Interaction Matrices
    max_residues = max(loaded_data['num_residues_list']) if loaded_data['num_residues_list'] else 0
    max_atoms = max(loaded_data['num_atoms_list']) if loaded_data['num_atoms_list'] else 0
    
    avg_grad_matrix = np.zeros((max_residues, max_atoms))
    avg_ig_matrix = np.zeros((max_residues, max_atoms))
    count_matrix = np.zeros((max_residues, max_atoms))

    for i in range(len(loaded_data['all_grad_atom_residue_matrices'])):
        grad_mat = loaded_data['all_grad_atom_residue_matrices'][i]
        ig_mat = loaded_data['all_ig_atom_residue_matrices'][i]
        
        padded_grad = np.pad(grad_mat, ((0, max_residues - grad_mat.shape[0]), (0, max_atoms - grad_mat.shape[1])))
        padded_ig = np.pad(ig_mat, ((0, max_residues - ig_mat.shape[0]), (0, max_atoms - ig_mat.shape[1])))

        avg_grad_matrix += padded_grad
        avg_ig_matrix += padded_ig
        count_matrix[:grad_mat.shape[0], :grad_mat.shape[1]] += 1
    
    avg_grad_matrix = np.divide(avg_grad_matrix, count_matrix, where=count_matrix!=0)
    avg_ig_matrix = np.divide(avg_ig_matrix, count_matrix, where=count_matrix!=0)

    # t-SNE embeddings
    protein_tsne = None
    ligand_tsne = None
    if loaded_data['all_protein_embeddings'] and loaded_data['all_ligand_embeddings']:
        protein_embeddings = np.vstack(loaded_data['all_protein_embeddings'])
        ligand_embeddings = np.vstack(loaded_data['all_ligand_embeddings'])
        protein_attrs = np.concatenate(loaded_data['all_protein_attributions'])
        ligand_attrs = np.concatenate(loaded_data['all_ligand_attributions'])

        if protein_embeddings.shape[0] > 1:
            perplexity = min(30, protein_embeddings.shape[0] - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            protein_tsne = tsne.fit_transform(protein_embeddings)

        if ligand_embeddings.shape[0] > 1:
            perplexity = min(30, ligand_embeddings.shape[0] - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            ligand_tsne = tsne.fit_transform(ligand_embeddings)

    return {
        "avg_grad_matrix": avg_grad_matrix,
        "avg_ig_matrix": avg_ig_matrix,
        "protein_tsne": protein_tsne,
        "ligand_tsne": ligand_tsne,
        "protein_attributions": protein_attrs if 'protein_attrs' in locals() else None,
        "ligand_attributions": ligand_attrs if 'ligand_attrs' in locals() else None,
    }
