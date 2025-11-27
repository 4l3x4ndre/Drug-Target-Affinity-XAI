"""
Advanced utilities for interaction analysis and validation.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt


class InteractionAnalyzer:
    """Advanced analysis tools for interaction explanations."""
    
    def __init__(self):
        pass
    
    def compare_methods(
        self,
        gradient_result: Dict,
        ig_result: Dict,
        top_k: int = 50
    ) -> Dict:
        """
        Comprehensive comparison between gradient and IG methods.
        
        Returns:
            Dictionary with correlation metrics, overlap statistics, etc.
        """
        grad_matrix = gradient_result['interaction_matrix'].flatten()
        ig_matrix = ig_result['interaction_matrix'].flatten()
        
        # Remove zeros and NaNs
        mask = ~(np.isnan(grad_matrix) | np.isnan(ig_matrix))
        grad_matrix = grad_matrix[mask]
        ig_matrix = ig_matrix[mask]
        
        # Correlation metrics
        pearson_corr = np.corrcoef(grad_matrix, ig_matrix)[0, 1]
        spearman_corr, _ = spearmanr(grad_matrix, ig_matrix)
        kendall_corr, _ = kendalltau(grad_matrix, ig_matrix)
        
        # Cosine similarity
        cosine_sim = 1 - cosine(np.abs(grad_matrix), np.abs(ig_matrix))
        
        # Top-k overlap
        grad_top_indices = np.argsort(np.abs(grad_matrix))[-top_k:]
        ig_top_indices = np.argsort(np.abs(ig_matrix))[-top_k:]
        overlap = len(set(grad_top_indices) & set(ig_top_indices))
        overlap_pct = (overlap / top_k) * 100
        
        # Sign agreement (do they agree on direction?)
        sign_agreement = np.mean(np.sign(grad_matrix) == np.sign(ig_matrix)) * 100
        
        # Magnitude comparison
        grad_mean = np.abs(grad_matrix).mean()
        ig_mean = np.abs(ig_matrix).mean()
        magnitude_ratio = grad_mean / (ig_mean + 1e-10)
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'kendall_tau': kendall_corr,
            'cosine_similarity': cosine_sim,
            'top_k_overlap': overlap,
            'top_k_overlap_pct': overlap_pct,
            'sign_agreement_pct': sign_agreement,
            'magnitude_ratio': magnitude_ratio,
            'gradient_mean_abs': grad_mean,
            'ig_mean_abs': ig_mean
        }
    
    def cluster_interactions(
        self,
        interaction_matrix: np.ndarray,
        protein_sequence: np.ndarray,
        n_clusters: int = 5,
        method: str = 'kmeans'
    ) -> Dict:
        """
        Cluster residues based on their interaction patterns with atoms.
        
        Args:
            interaction_matrix: [num_residues, num_atoms]
            protein_sequence: Protein sequence tensor
            n_clusters: Number of clusters
            method: 'kmeans' or 'hierarchical'
            
        Returns:
            Dictionary with cluster assignments and statistics
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        # Filter to non-padding residues
        seq_np = protein_sequence.detach().cpu().numpy().flatten()
        non_padding_mask = seq_np != 0
        filtered_matrix = interaction_matrix[non_padding_mask, :]
        
        # Cluster residues based on interaction patterns
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clusterer.fit_predict(filtered_matrix)
        
        # Analyze clusters
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_interactions = filtered_matrix[cluster_mask]
            
            cluster_stats.append({
                'cluster_id': i,
                'size': np.sum(cluster_mask),
                'mean_interaction': cluster_interactions.mean(),
                'max_interaction': cluster_interactions.max(),
                'residue_indices': np.where(non_padding_mask)[0][cluster_mask].tolist()
            })
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_stats,
            'n_clusters': n_clusters,
            'method': method
        }
    
    def analyze_interaction_sparsity(
        self,
        interaction_matrix: np.ndarray,
        thresholds: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict:
        """
        Analyze sparsity of interaction matrix at different thresholds.
        """
        results = {}
        total_interactions = interaction_matrix.size
        
        for threshold in thresholds:
            significant = np.abs(interaction_matrix) > threshold
            n_significant = np.sum(significant)
            sparsity = 1 - (n_significant / total_interactions)
            
            results[f'threshold_{threshold}'] = {
                'n_significant': int(n_significant),
                'percentage': (n_significant / total_interactions) * 100,
                'sparsity': sparsity * 100
            }
        
        return results
    
    def compute_interaction_statistics(
        self,
        interaction_matrix: np.ndarray,
        protein_sequence: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive statistics about the interaction matrix.
        """
        # Filter padding
        seq_np = protein_sequence.detach().cpu().numpy().flatten()
        non_padding_mask = seq_np != 0
        filtered_matrix = interaction_matrix[non_padding_mask, :]
        
        stats = {
            'shape': filtered_matrix.shape,
            'total_interactions': filtered_matrix.size,
            'mean': float(filtered_matrix.mean()),
            'std': float(filtered_matrix.std()),
            'min': float(filtered_matrix.min()),
            'max': float(filtered_matrix.max()),
            'mean_abs': float(np.abs(filtered_matrix).mean()),
            'median': float(np.median(filtered_matrix)),
            'median_abs': float(np.median(np.abs(filtered_matrix))),
            'percentile_95': float(np.percentile(np.abs(filtered_matrix), 95)),
            'percentile_99': float(np.percentile(np.abs(filtered_matrix), 99)),
            'n_positive': int(np.sum(filtered_matrix > 0)),
            'n_negative': int(np.sum(filtered_matrix < 0)),
            'n_zero': int(np.sum(filtered_matrix == 0)),
        }
        
        # Compute entropy (measure of interaction diversity)
        abs_matrix = np.abs(filtered_matrix)
        normalized = abs_matrix / (abs_matrix.sum() + 1e-10)
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        stats['entropy'] = float(entropy)
        
        return stats
    
    def identify_binding_pockets(
        self,
        interaction_matrix: np.ndarray,
        protein_sequence: np.ndarray,
        window_size: int = 10,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Identify potential binding pockets based on interaction patterns.
        
        Uses a sliding window to find regions with high cumulative interactions.
        """
        seq_np = protein_sequence.detach().cpu().numpy().flatten()
        non_padding_mask = seq_np != 0
        filtered_matrix = interaction_matrix[non_padding_mask, :]
        
        # Compute residue importance (sum across atoms)
        residue_importance = np.sum(np.abs(filtered_matrix), axis=1)
        
        # Sliding window to find pockets
        pockets = []
        seq_length = len(residue_importance)
        
        for i in range(seq_length - window_size + 1):
            window_score = np.sum(residue_importance[i:i+window_size])
            pockets.append({
                'start_idx': int(np.where(non_padding_mask)[0][i]),
                'end_idx': int(np.where(non_padding_mask)[0][i + window_size - 1]),
                'score': float(window_score),
                'mean_score': float(window_score / window_size)
            })
        
        # Sort by score and return top-k
        pockets_sorted = sorted(pockets, key=lambda x: x['score'], reverse=True)
        return pockets_sorted[:top_k]
    
    def compute_atom_residue_distances(
        self,
        interaction_matrix: np.ndarray,
        mol_obj,
        protein_sequence: np.ndarray,
        protein_coords: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        If 3D coordinates are available, compute spatial distances and
        correlate with interaction scores.
        
        Note: This requires 3D coordinates which may not always be available.
        """
        if protein_coords is None:
            return None
        
        # Get ligand coordinates from RDKit
        from rdkit.Chem import AllChem
        
        if mol_obj.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_obj)
        
        conf = mol_obj.GetConformer()
        ligand_coords = np.array([
            [conf.GetAtomPosition(i).x, 
             conf.GetAtomPosition(i).y, 
             conf.GetAtomPosition(i).z]
            for i in range(mol_obj.GetNumAtoms())
        ])
        
        # Filter protein coordinates to non-padding residues
        seq_np = protein_sequence.detach().cpu().numpy().flatten()
        non_padding_mask = seq_np != 0
        filtered_protein_coords = protein_coords[non_padding_mask]
        
        # Compute distance matrix
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(filtered_protein_coords, ligand_coords)
        
        # Correlate with interaction scores
        filtered_interactions = interaction_matrix[non_padding_mask, :]
        
        # Flatten both matrices
        dist_flat = distance_matrix.flatten()
        inter_flat = np.abs(filtered_interactions).flatten()
        
        # Compute correlation (expect negative: closer = stronger interaction)
        correlation, p_value = spearmanr(dist_flat, inter_flat)
        
        return {
            'distance_matrix': distance_matrix,
            'correlation_with_interactions': correlation,
            'p_value': p_value,
            'mean_distance': float(distance_matrix.mean()),
            'min_distance': float(distance_matrix.min())
        }


def plot_method_comparison(comparison_results: Dict) -> go.Figure:
    """
    Visualize comparison between gradient and IG methods.
    """
    metrics = [
        'pearson_correlation',
        'spearman_correlation', 
        'kendall_tau',
        'cosine_similarity'
    ]
    values = [comparison_results[m] for m in metrics]
    
    fig = go.Figure(go.Bar(
        x=metrics,
        y=values,
        marker_color=['blue' if v > 0.7 else 'orange' if v > 0.5 else 'red' 
                     for v in values],
        text=[f'{v:.3f}' for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Method Comparison: Correlation Metrics',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        height=400,
        showlegend=False
    )
    
    # Add threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                  annotation_text="Good (>0.7)")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="Fair (>0.5)")
    
    return fig


def plot_sparsity_analysis(sparsity_results: Dict) -> go.Figure:
    """
    Visualize interaction sparsity at different thresholds.
    """
    thresholds = []
    sparsities = []
    n_significant = []
    
    for key, value in sparsity_results.items():
        threshold = float(key.split('_')[1])
        thresholds.append(threshold)
        sparsities.append(value['sparsity'])
        n_significant.append(value['n_significant'])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sparsity vs Threshold', 'Number of Significant Interactions')
    )
    
    # Sparsity plot
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=sparsities,
            mode='lines+markers',
            name='Sparsity',
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    
    # Significant interactions plot
    fig.add_trace(
        go.Bar(
            x=[str(t) for t in thresholds],
            y=n_significant,
            name='Significant',
            marker_color='green'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Sparsity (%)", row=1, col=1)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_layout(height=400, width=1000, showlegend=False)
    
    return fig


def plot_binding_pockets(
    pockets: List[Dict],
    protein_sequence: np.ndarray,
    int2vocab: Dict
) -> go.Figure:
    """
    Visualize identified binding pockets along the protein sequence.
    """
    seq_np = protein_sequence.detach().cpu().numpy().flatten()
    
    fig = go.Figure()
    
    # Plot protein sequence as baseline
    fig.add_trace(go.Scatter(
        x=list(range(len(seq_np))),
        y=[0] * len(seq_np),
        mode='lines',
        line=dict(color='gray', width=1),
        name='Protein Sequence',
        showlegend=False
    ))
    
    # Highlight pockets
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, pocket in enumerate(pockets):
        start = pocket['start_idx']
        end = pocket['end_idx']
        score = pocket['score']
        
        # Create sequence string for this pocket
        pocket_seq = ''.join([
            int2vocab.get(str(int(seq_np[j])), '?')
            for j in range(start, end + 1)
        ])
        
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=colors[i % len(colors)],
            opacity=0.3,
            layer="below",
            line_width=0,
            annotation_text=f"Pocket {i+1}<br>Score: {score:.2f}",
            annotation_position="top"
        )
        
        # Add label
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[i+1, i+1],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=4),
            name=f'Pocket {i+1}: {start}-{end}',
            hovertemplate=f'Pocket {i+1}<br>Residues: {start}-{end}<br>Sequence: {pocket_seq}<br>Score: {score:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Identified Binding Pockets',
        xaxis_title='Residue Position',
        yaxis_title='Pocket',
        height=400,
        width=1000
    )
    
    return fig


def create_statistics_dashboard(stats: Dict) -> go.Figure:
    """
    Create a dashboard showing interaction statistics.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribution Statistics',
            'Sign Distribution',
            'Percentile Analysis',
            'Entropy & Sparsity'
        ),
        specs=[[{'type': 'bar'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'indicator'}]]
    )
    
    # Distribution statistics
    metrics = ['mean', 'median', 'std', 'mean_abs', 'median_abs']
    values = [stats[m] for m in metrics]
    
    fig.add_trace(
        go.Bar(x=metrics, y=values, name='Stats'),
        row=1, col=1
    )
    
    # Sign distribution
    fig.add_trace(
        go.Pie(
            labels=['Positive', 'Negative', 'Zero'],
            values=[stats['n_positive'], stats['n_negative'], stats['n_zero']],
            hole=0.3
        ),
        row=1, col=2
    )
    
    # Percentiles
    fig.add_trace(
        go.Bar(
            x=['95th', '99th', 'Max'],
            y=[stats['percentile_95'], stats['percentile_99'], stats['max']],
            name='Percentiles'
        ),
        row=2, col=1
    )
    
    # Entropy indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=stats['entropy'],
            title={'text': "Entropy"},
            gauge={'axis': {'range': [0, 20]}}
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1000, showlegend=False)
    
    return fig


#  usage function
def run_comprehensive_analysis(
    gradient_result: Dict,
    ig_result: Dict,
    protein_sequence: np.ndarray,
    mol_obj,
    int2vocab: Dict
) -> Dict:
    """
    Run all analyses and return results with figures.
    """
    analyzer = InteractionAnalyzer()
    
    # Compare methods
    comparison = analyzer.compare_methods(gradient_result, ig_result, top_k=50)
    fig_comparison = plot_method_comparison(comparison)
    
    # Sparsity analysis
    sparsity = analyzer.analyze_interaction_sparsity(
        gradient_result['interaction_matrix']
    )
    fig_sparsity = plot_sparsity_analysis(sparsity)
    
    # Statistics
    stats = analyzer.compute_interaction_statistics(
        gradient_result['interaction_matrix'],
        protein_sequence
    )
    fig_stats = create_statistics_dashboard(stats)
    
    # Binding pockets
    pockets = analyzer.identify_binding_pockets(
        gradient_result['interaction_matrix'],
        protein_sequence,
        window_size=10,
        top_k=5
    )
    fig_pockets = plot_binding_pockets(pockets, protein_sequence, int2vocab)
    
    # Cluster analysis
    clusters = analyzer.cluster_interactions(
        gradient_result['interaction_matrix'],
        protein_sequence,
        n_clusters=5
    )
    
    return {
        'comparison': comparison,
        'sparsity': sparsity,
        'statistics': stats,
        'pockets': pockets,
        'clusters': clusters,
        'figures': {
            'comparison': fig_comparison,
            'sparsity': fig_sparsity,
            'statistics': fig_stats,
            'pockets': fig_pockets
        }
    }
