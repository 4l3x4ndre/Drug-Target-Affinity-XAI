import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr, wasserstein_distance, kendalltau
from xai.interaction_explainer import InteractionExplainer
from xai.explainer import Explainer
from MGraphDTA.regression.model import MGraphDTA
from datetime import datetime

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_top_k_indices(matrix, k):
    return np.argsort(np.abs(matrix.flatten()))[-k:]

def calculate_centroid(matrix):
    """Calculate the center of mass (centroid) of a 2D matrix."""
    matrix = np.abs(matrix)
    if np.sum(matrix) == 0:
        return np.array([0, 0])
    matrix = matrix / np.sum(matrix)
    y_coords, x_coords = np.indices(matrix.shape)
    centroid_y = np.sum(y_coords * matrix)
    centroid_x = np.sum(x_coords * matrix)
    return np.array([centroid_y, centroid_x])

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors, handling zero vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def load_model_for_analysis(device='cpu', saved_model_path:str='', dataset_name='kiba') -> Explainer:
    """Load model once for analysis."""
    model_config = {
        'block_num': 3,
        'vocab_protein_size': 25 + 1,
        'embedding_size': 128,
        'filter_num': 32,
        'out_dim': 1
    }
    
    model = MGraphDTA(**model_config).to(device)
    if saved_model_path:
        model_path = saved_model_path
    else:
        model_path = os.path.join(os.getcwd(), "models", dataset_name, 
                                  "fold-0, repeat-0, epoch-2320, train_loss-0.0028, train_cindex-0.9946, val_loss-0.1456, val_cindex-0.8923, val_r2-0.7678.pt")
    
    model_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(model_dict)
    model.eval()
    
    explainer = Explainer(model, dataset_name=dataset_name, device=device)
    return explainer

def run_analysis(device, saved_model, num_samples, k_top_interactions, top_x_for_correlation, ig_steps, output_dir, batch_size):
    set_seed(42)
    
    print("Loading model and explainer...")
    explainer = load_model_for_analysis(device, saved_model_path=saved_model, dataset_name='kiba')
    interaction_explainer = InteractionExplainer(explainer.model, device=device)

    subfolder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"run_{subfolder_name}")
    os.makedirs(results_dir, exist_ok=True)

    metadata = {
        "num_samples": num_samples, "k_top_interactions": k_top_interactions,
        "top_x_for_correlation": top_x_for_correlation, "ig_steps": ig_steps,
        "batch_size": batch_size, "model_path": saved_model
    }
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    csv_path = os.path.join(results_dir, "results.csv")
    is_first_batch = True

    num_residues_list, num_atoms_list = [], []
    all_top_protein_features, all_top_ligand_features, all_top_interaction_scores = [], [], []

    for batch_num, i in enumerate(range(0, num_samples, batch_size)):
        start_index = i
        end_index = min(i + batch_size, num_samples)
        print(f"Processing batch {batch_num + 1}/{int(np.ceil(num_samples/batch_size))}: samples {start_index} to {end_index - 1}")

        batch_results = []
        batch_grad_matrices, batch_ig_matrices = [], []
        batch_prot_embeds, batch_lig_embeds = [], []
        batch_prot_attrs, batch_lig_attrs = [], []
        batch_sample_indices = []

        for sample_idx in tqdm(range(start_index, end_index), desc=f"Batch {batch_num + 1}"):
            smiles, protein = explainer.get_data(sample_idx)
            data, (g, mol) = explainer.dataset.transform_unique(smiles, protein)
            data = data.to(device)
            
            grad_res = interaction_explainer.gradient_atom_residue_map(data, data.target)
            ig_res = interaction_explainer.integrated_gradients_interaction(data, data.target, n_steps=ig_steps)
            
            mask = grad_res['non_padding_mask']
            grad_matrix = grad_res['interaction_matrix'][mask, :]
            ig_matrix = ig_res['interaction_matrix'][mask, :]

            batch_grad_matrices.append(grad_matrix)
            batch_ig_matrices.append(ig_matrix)
            num_residues_list.append(grad_matrix.shape[0])
            num_atoms_list.append(grad_matrix.shape[1])

            grad_flat = grad_matrix.flatten()
            ig_flat = ig_matrix.flatten()
            min_len = min(len(grad_flat), len(ig_flat))
            grad_flat, ig_flat = grad_flat[:min_len], ig_flat[:min_len]

            pearson_corr, spearman_corr, kendall_tau, iou, cos_sim, sign_agreement_perc = [np.nan] * 6
            if len(grad_flat) > 1:
                top_x_indices = np.argsort(np.abs(grad_flat))[-top_x_for_correlation:]
                filtered_grad = grad_flat[top_x_indices]
                filtered_ig = ig_flat[top_x_indices]
                if len(np.unique(filtered_grad)) > 1 and len(np.unique(filtered_ig)) > 1:
                    pearson_corr = np.corrcoef(filtered_grad, filtered_ig)[0, 1]
                    spearman_corr, _ = spearmanr(filtered_grad, filtered_ig)
                    kendall_tau, _ = kendalltau(filtered_grad, filtered_ig)
                
                top_k_grad_indices = get_top_k_indices(grad_matrix, k_top_interactions)
                top_k_ig_indices = get_top_k_indices(ig_matrix, k_top_interactions)
                iou = len(np.intersect1d(top_k_grad_indices, top_k_ig_indices)) / len(np.union1d(top_k_grad_indices, top_k_ig_indices))
                cos_sim = cosine_similarity(grad_flat, ig_flat)
                sign_agreement_perc = np.sum(np.sign(grad_flat) == np.sign(ig_flat)) / len(grad_flat) * 100

            mae = np.mean(np.abs(grad_flat - ig_flat))
            wasserstein_dist = wasserstein_distance(np.abs(grad_flat), np.abs(ig_flat))
            centroid_dist = np.linalg.norm(calculate_centroid(grad_matrix) - calculate_centroid(ig_matrix))
            
            batch_results.append({
                "Pearson": pearson_corr, "Spearman": spearman_corr, "Kendall_Tau": kendall_tau,
                "IoU": iou, "Cosine_Sim": cos_sim, "Sign_Agree%": sign_agreement_perc,
                "MAE": mae, "Wasserstein": wasserstein_dist, "Centroid_Dist": centroid_dist
            })

            feature_res = interaction_explainer.gradient_interaction_map(data)
            batch_prot_embeds.append(feature_res['protein_features'])
            batch_lig_embeds.append(feature_res['ligand_features'])
            batch_prot_attrs.append(feature_res['protein_attributions'])
            batch_lig_attrs.append(feature_res['ligand_attributions'])
            
            top_indices = np.unravel_index(np.argsort(np.abs(feature_res['interaction_map'].flatten()))[-k_top_interactions:], feature_res['interaction_map'].shape)
            all_top_protein_features.extend(top_indices[0])
            all_top_ligand_features.extend(top_indices[1])
            all_top_interaction_scores.extend(feature_res['interaction_map'][top_indices])

        pd.DataFrame(batch_results).to_csv(csv_path, mode='a', header=is_first_batch, index=False)
        is_first_batch = False
        
        np.savez_compressed(os.path.join(results_dir, f"grad_matrices_batch_{batch_num}.npz"), *batch_grad_matrices)
        np.savez_compressed(os.path.join(results_dir, f"ig_matrices_batch_{batch_num}.npz"), *batch_ig_matrices)
        np.savez_compressed(os.path.join(results_dir, f"protein_embeddings_batch_{batch_num}.npz"), *batch_prot_embeds)
        np.savez_compressed(os.path.join(results_dir, f"ligand_embeddings_batch_{batch_num}.npz"), *batch_lig_embeds)
        np.savez_compressed(os.path.join(results_dir, f"protein_attributions_batch_{batch_num}.npz"), *batch_prot_attrs)
        np.savez_compressed(os.path.join(results_dir, f"ligand_attributions_batch_{batch_num}.npz"), *batch_lig_attrs)

    np.save(os.path.join(results_dir, "num_residues_list.npy"), np.array(num_residues_list))
    np.save(os.path.join(results_dir, "num_atoms_list.npy"), np.array(num_atoms_list))
    np.savez(os.path.join(results_dir, "top_features.npz"),
             protein_features=all_top_protein_features,
             ligand_features=all_top_ligand_features,
             interaction_scores=all_top_interaction_scores)

    print(f"Analysis complete. Results saved to {results_dir}")
