import numpy as np
import torch
from typing import Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from src.utils.plot_utils import get_styled_figure_ax


class InteractionExplainer:
    """
    Explains drug-target interactions by attributing the affinity score to 
    individual atom-residue pairs using gradient-based and integrated gradient methods.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def gradient_interaction_map(self, data) -> Dict:
        """
        Compute interaction map using gradients of prediction w.r.t. 
        concatenated features (simple gradient method).
        
        Returns:
            Dictionary containing:
            - interaction_map: [num_residues x num_atoms] matrix
            - protein_attributions: [num_residues] importance scores
            - ligand_attributions: [num_atoms] importance scores
        """
        self.model.zero_grad()
        
        # Forward pass to get intermediate features
        target = data.target.to(self.device)
        # target.requires_grad = True
        
        # Get protein embeddings (after encoding)
        protein_x = self.model.protein_encoder(target)  # [batch, 96]
        
        # Get ligand embeddings (after graph encoding)
        ligand_x = self.model.ligand_encoder(data)  # [batch, 96]
        
        # Make sure they require gradients
        # protein_x.requires_grad = True
        # ligand_x.requires_grad = True
        protein_x.retain_grad()
        ligand_x.retain_grad()
        
        # Concatenate and pass through classifier
        concat_features = torch.cat([protein_x, ligand_x], dim=-1)
        prediction = self.model.classifier(concat_features)
        
        # Compute gradients w.r.t. the concatenated features
        prediction.backward()
        
        # Get gradients
        protein_grad = protein_x.grad.detach().cpu().numpy()[0]  # [96]
        ligand_grad = ligand_x.grad.detach().cpu().numpy()[0]    # [96]
        
        # Get the actual feature values
        protein_features = protein_x.detach().cpu().numpy()[0]
        ligand_features = ligand_x.detach().cpu().numpy()[0]
        
        # Compute attributions (gradient × input)
        protein_attribution = protein_grad * protein_features  # [96]
        ligand_attribution = ligand_grad * ligand_features    # [96]
        
        # Compute interaction map as outer product of attributions
        # This gives us a [96 x 96] matrix showing feature-level interactions
        interaction_map = np.outer(protein_attribution, ligand_attribution)
        
        return {
            'interaction_map': interaction_map,
            'protein_attributions': protein_attribution,
            'ligand_attributions': ligand_attribution,
            'protein_features': protein_features,
            'ligand_features': ligand_features,
            'prediction': prediction.item()
        }
    
    def gradient_atom_residue_map(self, data, protein_sequence) -> Dict:
        """
        Compute atom-to-residue interaction map by tracing gradients back to 
        raw inputs (atoms and residues).
        
        This method computes:
        1. Gradients w.r.t. protein sequence embeddings -> residue attributions
        2. Gradients w.r.t. node features -> atom attributions
        3. Outer product to create interaction matrix
        """
        self.model.zero_grad()
        
        # Clone data to avoid modifying original
        data_copy = data.clone()
        
        # Store original node features and enable gradient tracking
        original_node_features = data_copy.x.clone()
        original_node_features.requires_grad = True
        data_copy.x = original_node_features
        
        # Prepare target with gradient tracking
        target = data_copy.target.to(self.device)
        
        # Get protein embeddings after initial embedding layer
        protein_embed = self.model.protein_encoder.embed(target)  # [batch, seq_len, embed_dim]
        protein_embed.retain_grad()
        
        # Continue through protein encoder
        protein_embed_permuted = protein_embed.permute(0, 2, 1)
        feats = [block(protein_embed_permuted) for block in self.model.protein_encoder.block_list]
        protein_x = torch.cat(feats, -1)
        protein_x = self.model.protein_encoder.linear(protein_x)
        
        # Get ligand features
        # Important: Use original node features, not the modified data.x
        ligand_x = self.model.ligand_encoder(data_copy)
        
        # Forward through classifier
        concat_features = torch.cat([protein_x, ligand_x], dim=-1)
        prediction = self.model.classifier(concat_features)
        
        # Backward pass
        prediction.backward()

        # Log shapes:
        logger.debug(f"protein_embed shape: {protein_embed.shape}")
        logger.debug(f"original_node_features shape: {original_node_features.shape}")
        
        # Get gradients w.r.t. protein embeddings (per residue)
        if protein_embed.grad is not None:
            protein_grad = protein_embed.grad.detach().cpu().numpy()[0]  # [seq_len, embed_dim]
            protein_embed_np = protein_embed.detach().cpu().numpy()[0]
            # Attribution per residue (sum across embedding dimension)
            residue_attributions = np.sum(protein_grad * protein_embed_np, axis=1)  # [seq_len]
        else:
            # Fallback if no gradient
            logger.warning("No gradient found for protein embeddings.")
            #residue_attributions = np.zeros(protein_embed.shape[1])
            raise ValueError("No gradient computed for protein embeddings.")
        
        # Get gradients w.r.t. node features (per atom)
        if original_node_features.grad is not None:
            atom_grad = original_node_features.grad.detach().cpu().numpy()  # [num_atoms, feature_dim]
            atom_features = original_node_features.detach().cpu().numpy()
            # Attribution per atom (sum across feature dimension)
            atom_attributions = np.sum(atom_grad * atom_features, axis=1)  # [num_atoms]
        else:
            # Fallback if no gradient
            logger.warning("No gradient found for node features.")
            #atom_attributions = np.zeros(original_node_features.shape[0])
            raise ValueError("No gradient computed for node features.")

        # log shapes:
        logger.debug(f"residue_attributions shape: {residue_attributions.shape}")
        logger.debug(f"atom_attributions shape: {atom_attributions.shape}")
        logger.debug(f"interaction matrix shape: {(len(residue_attributions), len(atom_attributions))}")
        
        # Create atom-to-residue interaction matrix
        # Shape: [num_residues, num_atoms]
        interaction_matrix = np.outer(residue_attributions, atom_attributions)
        
        # Normalize for visualization
        interaction_matrix_normalized = self._normalize_interaction_matrix(interaction_matrix)
        
        # Filter to non-padding residues
        seq_np = protein_sequence.detach().cpu().numpy().flatten()
        non_padding_mask = seq_np != 0
        
        return {
            'interaction_matrix': interaction_matrix,
            'interaction_matrix_normalized': interaction_matrix_normalized,
            'residue_attributions': residue_attributions,
            'atom_attributions': atom_attributions,
            'non_padding_mask': non_padding_mask,
            'num_atoms': len(atom_attributions),
            'num_residues': len(residue_attributions),
            'prediction': prediction.item()
        }
    
    def integrated_gradients_interaction(
        self, 
        data, 
        protein_sequence,
        n_steps: int = 50
    ) -> Dict:
        """
        Compute atom-to-residue interaction map using Integrated Gradients.
        
        Integrated Gradients accumulates gradients along a path from baseline to input,
        providing more stable and theoretically grounded attributions.
        """
        # === 1. Define Baselines ===
        baseline_data = data.clone()
        baseline_data.x = torch.zeros_like(data.x)
        baseline_target = torch.zeros_like(data.target).to(self.device)

        # === 2. Get Input and Baseline Embeddings (outside the loop) ===
        protein_embed_input = self.model.protein_encoder.embed(data.target.to(self.device)).detach()
        protein_embed_baseline = self.model.protein_encoder.embed(baseline_target).detach()
        
        # We also need the original node features
        node_features_input = data.x.detach()
        node_features_baseline = baseline_data.x.detach()

        # Storage for accumulated gradients
        acc_atom_grad = torch.zeros_like(node_features_input)
        acc_protein_grad = torch.zeros_like(protein_embed_input)

        # Interpolate between baseline and actual input
        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            
            # === 3. Interpolate in Embedding / Feature Space ===
            
            # A) Interpolate Ligand Features
            interp_node_features = node_features_baseline + alpha * (node_features_input - node_features_baseline)
            interp_node_features.requires_grad = True
            
            # Create a copy of the data to hold interpolated features
            interp_data = data.clone()
            interp_data.x = interp_node_features
            
            # B) Interpolate Protein Embeddings
            interp_protein_embed = protein_embed_baseline + alpha * (protein_embed_input - protein_embed_baseline)
            interp_protein_embed.requires_grad = True # Use requires_grad for this new leaf tensor

            # === 4. Forward Pass with Interpolated Inputs ===
            
            # A) Protein: Pass interpolated embedding *directly* to the blocks
            # (SKIP the .embed() layer)
            protein_embed_permuted = interp_protein_embed.permute(0, 2, 1)
            feats = [block(protein_embed_permuted) for block in self.model.protein_encoder.block_list]
            protein_x = torch.cat(feats, -1)
            protein_x = self.model.protein_encoder.linear(protein_x)
            
            # B) Ligand: Pass data with interpolated node features
            ligand_x = self.model.ligand_encoder(interp_data)
            
            # C) Classifier
            concat_features = torch.cat([protein_x, ligand_x], dim=-1)
            prediction = self.model.classifier(concat_features)
            
            # === 5. Backward Pass ===
            self.model.zero_grad()
            prediction.backward()
            
            # === 6. Accumulate Gradients ===
            if interp_node_features.grad is not None:
                acc_atom_grad += interp_node_features.grad.detach().cpu()
            
            if interp_protein_embed.grad is not None:
                acc_protein_grad += interp_protein_embed.grad.detach().cpu()
        
        # === 7. Average and Scale Attributions ===
        
        # Average gradients
        avg_atom_grad = acc_atom_grad / n_steps
        avg_protein_grad = acc_protein_grad / n_steps

        # Scale by (input - baseline)
        atom_diff = (node_features_input - node_features_baseline).cpu()
        atom_attributions = torch.sum(avg_atom_grad * atom_diff, axis=1).numpy()
        
        protein_embed_diff = (protein_embed_input - protein_embed_baseline).cpu()
        residue_attributions = torch.sum(avg_protein_grad * protein_embed_diff, axis=2).numpy()[0]
        
        interaction_matrix = np.outer(residue_attributions, atom_attributions)
        interaction_matrix_normalized = self._normalize_interaction_matrix(interaction_matrix)
        
        seq_np = protein_sequence.detach().cpu().numpy().flatten()
        non_padding_mask = seq_np != 0
        
        return {
            'interaction_matrix': interaction_matrix,
            'interaction_matrix_normalized': interaction_matrix_normalized,
            'residue_attributions': residue_attributions,
            'atom_attributions': atom_attributions,
            'non_padding_mask': non_padding_mask,
            'num_atoms': len(atom_attributions),
            'num_residues': len(residue_attributions),
            'method': 'integrated_gradients'
        }
    
    def _normalize_interaction_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize interaction matrix for better visualization."""
        # Use symmetric normalization
        abs_max = np.abs(matrix).max()
        if abs_max > 0:
            return matrix / abs_max
        return matrix
    
    def get_top_interactions(
        self, 
        interaction_matrix: np.ndarray,
        protein_sequence: np.ndarray,
        mol_obj,
        top_k: int = 20,
        int2vocab: Dict = None
    ) -> pd.DataFrame:
        """
        Extract top-k atom-residue interactions from the interaction matrix.
        
        Returns:
            DataFrame with columns: residue_idx, atom_idx, interaction_score, 
                                   residue_aa, atom_symbol
        """
        # Filter to non-padding residues
        seq_np = protein_sequence.detach().cpu().numpy().flatten()
        non_padding_indices = np.where(seq_np != 0)[0]
        
        # Filter matrix to non-padding residues only
        filtered_matrix = interaction_matrix[non_padding_indices, :]
        
        # Get top-k interactions
        flat_indices = np.argsort(np.abs(filtered_matrix).flatten())[-top_k:][::-1]
        residue_indices, atom_indices = np.unravel_index(
            flat_indices, 
            filtered_matrix.shape
        )
        
        # Map back to original residue indices
        residue_indices = non_padding_indices[residue_indices]
        
        # Extract amino acid information
        if int2vocab is None:
            int2vocab = {str(i): f"AA{i}" for i in range(26)}
            int2vocab['0'] = '-'
        
        interactions = []
        for res_idx, atom_idx in zip(residue_indices, atom_indices):
            score = interaction_matrix[res_idx, atom_idx]
            aa_code = int2vocab.get(str(int(seq_np[res_idx])), '?')
            
            # Get atom information from molecule
            if mol_obj is not None and atom_idx < mol_obj.GetNumAtoms():
                atom = mol_obj.GetAtomWithIdx(int(atom_idx))
                atom_symbol = atom.GetSymbol()
            else:
                atom_symbol = f"Atom{atom_idx}"
            
            interactions.append({
                'residue_idx': int(res_idx),
                'atom_idx': int(atom_idx),
                'interaction_score': float(score),
                'residue_aa': aa_code,
                'atom_symbol': atom_symbol
            })
        
        return pd.DataFrame(interactions)


def visualize_interaction_matrix(
    interaction_result: Dict,
    protein_sequence,
    mol_obj,
    int2vocab: Dict = None,
    title: str = "Atom-to-Residue Interaction Map",
    max_residues: int = 100,
    figsize: Tuple[int, int] = (10, 20)
) -> plt.Figure:
    """
    Visualize the atom-to-residue interaction matrix as a heatmap using Matplotlib.
    """
    interaction_matrix = interaction_result['interaction_matrix_normalized']
    non_padding_mask = interaction_result['non_padding_mask']
    
    # Filter to non-padding residues
    filtered_matrix = interaction_matrix[non_padding_mask, :]
    
    # Limit to max_residues for visualization
    if filtered_matrix.shape[0] > max_residues:
        filtered_matrix = filtered_matrix[:max_residues, :]
    
    # Create labels
    seq_np = protein_sequence.detach().cpu().numpy().flatten()
    if int2vocab is None:
        int2vocab = {str(i): f"AA{i}" for i in range(26)}
        int2vocab['0'] = '-'
    
    # Residue labels
    residue_labels = [
        f"{i}: {int2vocab.get(str(int(seq_np[i])), '?')}" 
        for i in range(min(len(seq_np[non_padding_mask]), max_residues))
    ]
    
    # Atom labels
    atom_labels = []
    if mol_obj is not None:
        for atom_idx in range(filtered_matrix.shape[1]):
            if atom_idx < mol_obj.GetNumAtoms():
                atom = mol_obj.GetAtomWithIdx(atom_idx)
                atom_labels.append(f"{atom_idx}: {atom.GetSymbol()}")
            else:
                atom_labels.append(f"{atom_idx}")
    else:
        atom_labels = [f"Atom {i}" for i in range(filtered_matrix.shape[1])]
    
    # Create matplotlib plot
    fig, ax = get_styled_figure_ax(figsize=figsize, aspect='auto', grid=False)
    im = ax.imshow(filtered_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, label="Normalized Interaction Score", shrink=0.5)

    ax.set_xticks(np.arange(len(atom_labels)))
    ax.set_xticklabels(atom_labels, rotation=90)
    ax.set_yticks(np.arange(len(residue_labels)))
    ax.set_yticklabels(residue_labels, fontsize=5)

    # ax.set_title(title)
    ax.set_xlabel("Drug Atoms")
    ax.set_ylabel("Protein Residues")
    
    fig.tight_layout()
    return fig


def visualize_marginal_attributions(
    interaction_result: Dict,
    protein_sequence,
    mol_obj,
    int2vocab: Dict = None,
    max_display: int = 30
) -> go.Figure:
    """
    Visualize marginal attributions (sum across atoms/residues) as bar charts.
    """
    residue_attr = interaction_result['residue_attributions']
    atom_attr = interaction_result['atom_attributions']
    non_padding_mask = interaction_result['non_padding_mask']
    
    # Filter residues
    residue_attr_filtered = residue_attr[non_padding_mask]
    
    # Get top-k for each
    top_residue_indices = np.argsort(np.abs(residue_attr_filtered))[-max_display:][::-1]
    top_atom_indices = np.argsort(np.abs(atom_attr))[-max_display:][::-1]
    
    # Create labels
    seq_np = protein_sequence.detach().cpu().numpy().flatten()
    seq_filtered = seq_np[non_padding_mask]
    
    if int2vocab is None:
        int2vocab = {str(i): f"AA{i}" for i in range(26)}
        int2vocab['0'] = '-'
    
    residue_labels = [
        f"Pos {np.where(non_padding_mask)[0][i]}: {int2vocab.get(str(int(seq_filtered[i])), '?')}" 
        for i in top_residue_indices
    ]
    
    if mol_obj is not None:
        atom_labels = []
        for atom_idx in top_atom_indices:
            if atom_idx < mol_obj.GetNumAtoms():
                atom = mol_obj.GetAtomWithIdx(int(atom_idx))
                atom_labels.append(f"Atom {atom_idx}: {atom.GetSymbol()}")
            else:
                atom_labels.append(f"Atom {atom_idx}")
    else:
        atom_labels = [f"Atom {i}" for i in top_atom_indices]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top Residue Attributions", "Top Atom Attributions")
    )
    
    # Residue attributions
    fig.add_trace(
        go.Bar(
            y=residue_labels,
            x=residue_attr_filtered[top_residue_indices],
            orientation='h',
            marker_color=['red' if x < 0 else 'blue' 
                         for x in residue_attr_filtered[top_residue_indices]],
            name='Residues'
        ),
        row=1, col=1
    )
    
    # Atom attributions
    fig.add_trace(
        go.Bar(
            y=atom_labels,
            x=atom_attr[top_atom_indices],
            orientation='h',
            marker_color=['red' if x < 0 else 'blue' 
                         for x in atom_attr[top_atom_indices]],
            name='Atoms'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=600,
        width=1200,
        showlegend=False,
        title_text="Marginal Attribution Scores"
    )
    
    fig.update_xaxes(title_text="Attribution Score", row=1, col=1)
    fig.update_xaxes(title_text="Attribution Score", row=1, col=2)
    
    return fig
