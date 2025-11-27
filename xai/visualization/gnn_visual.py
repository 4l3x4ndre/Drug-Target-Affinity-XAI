import plotly.graph_objects as go
import networkx as nx
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.utils.plot_utils import get_styled_figure_ax
DATASET_COLORS = ['#95BB63', '#BCBCE0', '#77b5b6', '#EA805D']


def get_gnn_explanation_figures(explanation, node_labels, features_labels, threshold):
    """
    Generate figures for GNNExplainer results.

    Args:
        explanation: PyG Explanation object (output of GNNExplainer)
        node_labels: list of node labels (e.g. ["C_0", "N_1", ...])
        features_labels: list of feature names (same order as node features)
        threshold (float): Edge importance threshold.
    
    Returns:
        (matplotlib.Figure, go.Figure): A tuple containing the feature importance figure and the graph figure.
    """
    fig_feat = None
    if explanation.node_mask is not None:
        feature_importance = explanation.node_mask.sum(dim=0).cpu().numpy()
        
        # Filter out zero-importance features
        non_zero_indices = np.where(feature_importance > 0)[0]
        if len(non_zero_indices) > 0:
            non_zero_values = feature_importance[non_zero_indices]
            non_zero_labels = [features_labels[i] for i in non_zero_indices]

            topk = min(15, len(non_zero_values))
            
            # Sort by importance and get top-k
            top_indices_in_non_zero = np.argsort(non_zero_values)[-topk:]
            
            # top_features = [non_zero_labels[i] for i in top_indices_in_non_zero]
            top_features = []
            for i in top_indices_in_non_zero:
                l = non_zero_labels[i]
                l = l.replace('_', ' ')
                if l == "Hybridization sp2":
                    l =  "Hybridization sp²"
                elif l == "Hybridization sp3":
                    l = "Hybridization sp³"
                top_features.append(l)
            top_values = non_zero_values[top_indices_in_non_zero]

            # Create matplotlib plot
            fig_feat, ax = get_styled_figure_ax(figsize=(13, topk * 0.5), aspect='none', grid=True)
            ax.barh(np.arange(len(top_features)), top_values, 
                    # color="lightgreen", 
                    color = DATASET_COLORS[0],
                    align='center')
            ax.set_yticks(np.arange(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel("Total Importance")
            ax.set_ylabel("Features")
            # ax.set_title("Top Feature Importances (GNNExplainer)")
            fig_feat.tight_layout()

    edge_index = explanation.edge_index.cpu().numpy()
    node_mask = explanation.node_mask.cpu() if explanation.node_mask is not None else torch.ones(len(node_labels), 1)
    edge_mask = explanation.edge_mask.cpu().numpy() if explanation.edge_mask is not None else np.ones(edge_index.shape[1])
    node_importance = node_mask.mean(dim=1).numpy()
    important_edge_indices = np.where(edge_mask > threshold)[0]
    
    fig_graph = None
    if len(important_edge_indices) > 0:
        subgraph_edge_index = edge_index[:, important_edge_indices]
        subgraph_edge_weights = edge_mask[important_edge_indices]
        G = nx.DiGraph() 
        for i in range(subgraph_edge_index.shape[1]):
            u, v = subgraph_edge_index[:, i]
            G.add_edge(int(u), int(v), weight=subgraph_edge_weights[i])

        if G.nodes:
            pos = nx.spring_layout(G, seed=42)
            fig_graph = go.Figure()
            weights = [data['weight'] for u, v, data in G.edges(data=True)]
            min_weight = min(weights) if weights else 0
            max_weight = max(weights) if weights else 1
            colormap = cm.get_cmap('bwr')
            arrow_annotations = []
            for u, v, data in G.edges(data=True):
                weight = data['weight']
                normalized_weight = (weight - min_weight) / (max_weight - min_weight) if (max_weight - min_weight) > 0 else 0.5
                rgb = colormap(normalized_weight)[:3]
                r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                color = f'rgba({r}, {g}, {b}, {0.2 + 0.8 * weight})'  
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                fig_graph.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], line=dict(width=1.0 + 4.0 * weight, color=color), hoverinfo='text', text=f'Importance: {weight:.3f}', mode='lines'))
                arrow_annotations.append(go.layout.Annotation(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y', showarrow=True, arrowhead=5, arrowsize=4, arrowwidth=1, arrowcolor=color))
            
            node_x, node_y, node_text, node_color_values = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node_labels[node])
                node_color_values.append(node_importance[node])
            
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                hovertemplate="<b>%{text}</b><br>Importance: %{customdata:.3f}<extra></extra>",
                customdata=node_color_values,
                marker=dict(
                    showscale=True, colorscale='GnBu', color=node_color_values, size=[20 + 30 * val for val in node_color_values],
                    colorbar=dict(thickness=15, title='Node Importance', xanchor='left'),
                    line_width=1.5, line_color='black'
                )
            )
            fig_graph.add_trace(node_trace)
            fig_graph.update_layout(
                title="Explanation Subgraph with edge importance threshold: {:.2f}".format(threshold),
                showlegend=False, annotations=arrow_annotations, hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white", height=600,
            )

    return fig_feat, fig_graph

def create_gnn_summary_cards(gnn_results):
    """
    Create summary cards for GNN explanation results.
    
    Args:
        gnn_results: Dictionary from explain_graph method
        
    Returns:
        Dictionary with summary statistics
    """
    node_mask = gnn_results['node_mask']
    edge_mask = gnn_results['edge_mask']
    top_nodes = gnn_results['top_nodes']
    top_edges = gnn_results['top_edges']
    
    # Calculate summary statistics
    if node_mask is not None:
        node_importance = node_mask.cpu().numpy()
        mean_node_importance = float(np.mean(node_importance))
        std_node_importance = float(np.std(node_importance))
        max_node_importance = float(np.max(node_importance))
        min_node_importance = float(np.min(node_importance))
    else:
        mean_node_importance = std_node_importance = max_node_importance = min_node_importance = 0.0
    
    if edge_mask is not None:
        edge_importance = edge_mask.cpu().numpy()
        mean_edge_importance = float(np.mean(edge_importance))
        std_edge_importance = float(np.std(edge_importance))
        max_edge_importance = float(np.max(edge_importance))
        min_edge_importance = float(np.min(edge_importance))
    else:
        mean_edge_importance = std_edge_importance = max_edge_importance = min_edge_importance = 0.0
    
    summary = {
        'prediction': gnn_results['original_prediction'],
        'num_atoms': len(node_importance) if node_mask is not None else 0,
        'num_bonds': len(edge_importance) if edge_mask is not None else 0,
        'mean_node_importance': mean_node_importance,
        'std_node_importance': std_node_importance,
        'max_node_importance': max_node_importance,
        'min_node_importance': min_node_importance,
        'mean_edge_importance': mean_edge_importance,
        'std_edge_importance': std_edge_importance,
        'max_edge_importance': max_edge_importance,
        'min_edge_importance': min_edge_importance,
        'top_node_index': top_nodes[0]['index'] if top_nodes else 0,
        'top_node_importance': top_nodes[0]['importance'] if top_nodes else 0.0,
        'top_edge_source': top_edges[0]['source'] if top_edges else 0,
        'top_edge_target': top_edges[0]['target'] if top_edges else 0,
        'top_edge_importance': top_edges[0]['importance'] if top_edges else 0.0
    }
    
    return summary



