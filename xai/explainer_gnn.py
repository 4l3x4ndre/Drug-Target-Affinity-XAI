import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import os
import pandas as pd
from loguru import logger
import plotly.graph_objects as go
import networkx as nx
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt

from MGraphDTA.regression.preprocessing import GNNDataset
from MGraphDTA.regression.model import MGraphDTA
from src.utils.plot_utils import get_styled_figure_ax


class GNNExplainerWrapper:
    """
    A wrapper class for GNN explainability that provides explanations for the molecular graph part
    of the MGraphDTA model using PyTorch Geometric's GNNExplainer.
    """
    
    def __init__(self, model, dataset_name='davis', device='cpu'):
        """
        Initialize the GNN explainer.
        
        Args:
            model: The trained MGraphDTA model
            dataset_name: Name of the dataset ('davis', 'full_toxcast', etc.)
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.dataset_name = dataset_name
        
        # Load dataset for data processing
        fpath = os.path.join("MGraphDTA/regression/data", dataset_name)
        self.dataset = GNNDataset(fpath, train=False, transform_unique=True)
        
        # Load test data
        test_df = pd.read_csv(os.path.join(fpath, 'raw', 'data_test.csv'))
        self.df = test_df
        
        if dataset_name == 'davis' or dataset_name == 'kiba':
            self.smile_list = list(test_df['compound_iso_smiles'].unique())
            self.target_list = list(test_df['target_sequence'].unique())
            self.labels = test_df['affinity'].values
        elif dataset_name == 'full_toxcast':
            self.smile_list = list(test_df['smiles'].unique())
            self.target_list = list(test_df['sequence'].unique())
            self.labels = test_df['label'].values
        
        # Initialize GNNExplainer with proper configuration
        from torch_geometric.explain.config import ModelConfig, ExplanationType
        from torch_geometric.explain import Explainer
        
        # Create model config
        model_config = ModelConfig(
            mode='regression',
            task_level='graph',
            return_type='raw'
        )
        
        # Initialize the explainer with configuration
        self.explainer = Explainer(
            model=None,  # Will be set during explanation
            algorithm=GNNExplainer(epochs=2000, lr=0.0001),
            explanation_type=ExplanationType.model,
            model_config=model_config,
            node_mask_type='object',
            edge_mask_type='object'
        )
        
        logger.info(f"GNNExplainer initialized for dataset: {dataset_name}")
    
    def get_data(self, idx):
        """Get SMILES and protein sequence for a given index."""
        return self.smile_list[idx], self.target_list[idx]
    
    def get_smile(self, idx):
        """Get SMILES string for a given index."""
        return self.smile_list[idx]
    
    def get_target(self, idx):
        """Get protein sequence for a given index."""
        return self.target_list[idx]
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.smile_list)
    
    def explain_graph(self, smiles_input, protein_input, data_set_idx=None):
        """
        Explain the molecular graph part of the model using GNNExplainer.
        
        Args:
            smiles_input: SMILES string for the ligand
            protein_input: Protein sequence string
            data_set_idx: Optional index for ground truth comparison
            
        Returns:
            Dictionary containing explanation results
        """
        # Transform inputs to model format
        data, gm = self.dataset.transform_unique(smiles_input, protein_input)
        g, mol = gm
        node_labels = [f"{atom.GetSymbol()}_{i}" for i, atom in enumerate(mol.GetAtoms())]
        data = data.to(self.device)
        
        # Create a wrapper model that only uses the ligand encoder
        class LigandOnlyModel(nn.Module):
            def __init__(self, full_model):
                super().__init__()
                self.ligand_encoder = full_model.ligand_encoder
                self.model = full_model
                
            def forward(self, x, edge_index, batch=None):
                # Create a data object with the graph structure
                data_obj = Data(x=x, edge_index=edge_index, batch=batch)
                # Get ligand representation
                ligand_x = self.ligand_encoder(data_obj)
                return ligand_x
        
        # Create the ligand-only model
        ligand_model = LigandOnlyModel(self.model).to(self.device)
        ligand_model.eval()
        
        # Prepare data for GNNExplainer
        x = data.x
        edge_index = data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        
        # Get original prediction
        with torch.no_grad():
            original_prediction = self.model(data).item()
        
        # Generate explanation
        try:
            # Set the model in the explainer
            self.explainer.model = ligand_model
            
            # Create a data object for the explainer
            data_obj = Data(x=x, edge_index=edge_index, batch=batch)
            
            explanation = self.explainer(
                x=x,
                edge_index=edge_index,
                batch=batch
            )
            
            # Extract node and edge importance
            node_mask = explanation.node_mask
            edge_mask = explanation.edge_mask
            
            # Get top important nodes and edges
            top_nodes = self._get_top_nodes(node_mask, node_labels, top_k=20)
            print("node_mask", node_mask)
            print("top_nodes", top_nodes)
            top_edges = self._get_top_edges(edge_mask, edge_index, node_labels, top_k=20)
            
            # Create molecular visualization with importance
            mol_visualization = self._create_molecular_visualization(
                mol, node_mask, edge_mask, smiles_input
            )
            
            # Create networkx graph for additional analysis
            nx_graph = self._create_networkx_graph(x, edge_index, node_mask, edge_mask, mol)
            
            result = {
                'original_prediction': original_prediction,
                'node_mask': node_mask,
                'edge_mask': edge_mask,
                'top_nodes': top_nodes,
                'top_edges': top_edges,
                'mol_visualization': mol_visualization,
                'nx_graph': nx_graph,
                'mol_obj': mol,
                'graph_obj': g,
                'smiles': smiles_input,
                'protein': protein_input
            }
            
            if data_set_idx is not None:
                result['true_affinity'] = self.labels[data_set_idx]
                if self.dataset_name == 'full_toxcast':
                    result['uniprot_id'] = self.df['Uniprot_ID'].values[data_set_idx]
            
            logger.info(f"GNN explanation completed. Prediction: {original_prediction:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in GNN explanation: {str(e)}")
            raise e

    def explain_attributes(self, smiles_input, protein_input, data_set_idx=None):
        """
        Placeholder for attribute explanation method.
        
        Args:
            smiles_input: SMILES string for the ligand
            protein_input: Protein sequence string
            data_set_idx: Optional index for ground truth comparison
            
        Returns:
            NotImplementedError
        """
        # Transform inputs to model format
        data, gm = self.dataset.transform_unique(smiles_input, protein_input)
        _, mol = gm
        node_labels = [f"{atom.GetSymbol()}_{i}" for i, atom in enumerate(mol.GetAtoms())]
        features_labels = [
            'H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I',  # One-hot atomic type (9 features)
            'Atomic_Number',  # Atomic number (1 feature)
            'Acceptor',  # Electron acceptor (1 feature)
            'Donor',  # Electron donor (1 feature)
            'Aromatic',  # Aromatic system (1 feature)
            'Hybridization_sp', 'Hybridization_sp2', 'Hybridization_sp3',  # Hybridization (3 features)
            'Num_Hydrogens',  # Number of connected hydrogens (1 feature)
            'Formal_Charge',  # Formal charge (1 feature)
            'Explicit_Valence',  # Explicit valence (1 feature)
            'Implicit_Valence',  # Implicit valence (1 feature)
            'Num_Explicit_Hs',  # Number of explicit Hs (1 feature)
            'Num_Radical_Electrons',  # Number of radical electrons (1 feature)
        ]
        

        data = data.to(self.device)
        # Prepare data for GNNExplainer
        x = data.x
        edge_index = data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        
        # Create a wrapper model that only uses the ligand encoder
        class LigandOnlyModel(nn.Module):
            def __init__(self, full_model):
                super().__init__()
                self.ligand_encoder = full_model.ligand_encoder
                self.model = full_model
                
            def forward(self, x, edge_index, batch=None):
                # Create a data object with the graph structure
                data_obj = Data(x=x, edge_index=edge_index, batch=batch)
                # Get ligand representation
                ligand_x = self.ligand_encoder(data_obj)
                return ligand_x
        
        ligand_model = LigandOnlyModel(self.model).to(self.device)
        ligand_model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_prediction = self.model(data).item()

        explainer = Explainer(
            model=ligand_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='graph',
                return_type='raw',
            ),
        ) 

        logger.debug("Number of nodes: {}".format(x.size(0)))
        logger.debug("Number of edges: {}".format(edge_index.size(1)))
        logger.debug("Node feature shape: {}".format(x.shape))
        logger.debug("Edge index shape: {}".format(edge_index.shape))
        explanation = explainer(x, edge_index)
        explanation.visualize_feature_importance(f"feature_importance.png", top_k=10,
                                                 feat_labels=features_labels)
        explanation.visualize_graph(f"subgraph.png", backend='graphviz',
                                    node_labels=node_labels)

        return explanation, node_labels, features_labels


    def _get_top_nodes(self, node_mask, node_labels, top_k=10):
        """Get top k most important nodes."""
        if node_mask is None:
            return []
        
        node_importance = node_mask.cpu().numpy()
        # Flatten if needed
        if node_importance.ndim > 1:
            node_importance = node_importance.flatten()
        
        # Get unique importance values and their indices
        unique_importance = np.unique(node_importance)
        if len(unique_importance) == 1:
            # If all nodes have the same importance, just return the first k nodes
            top_indices = list(range(min(top_k, len(node_importance))))
        else:
            top_indices = np.argsort(node_importance)[-top_k:][::-1]
        
        top_nodes = []
        for idx in top_indices:
            top_nodes.append({
                'index': int(idx),
                'importance': float(node_importance[idx]),
                'label': node_labels[idx]
            })
        
        return top_nodes
    
    def _get_top_edges(self, edge_mask, edge_index, node_labels, top_k=10):
        """Get top k most important edges."""
        if edge_mask is None:
            return []
        
        edge_importance = edge_mask.cpu().numpy()
        # Flatten if needed
        if edge_importance.ndim > 1:
            edge_importance = edge_importance.flatten()
        
        # Get unique importance values
        unique_importance = np.unique(edge_importance)
        if len(unique_importance) == 1:
            # If all edges have the same importance, just return the first k edges
            top_indices = list(range(min(top_k, len(edge_importance))))
        else:
            top_indices = np.argsort(edge_importance)[-top_k:][::-1]
        
        top_edges = []
        for idx in top_indices:
            src, dst = edge_index[:, idx].cpu().numpy()
            top_edges.append({
                'source': node_labels[int(src)],
                'target': node_labels[int(dst)],
                'importance': float(edge_importance[idx])
            })
        
        return top_edges
    
    def _create_molecular_visualization(self, mol, node_mask, edge_mask, smiles):
        """Create molecular visualization with importance highlighting as SVG."""
        # Normalize node importance for coloring
        if node_mask is not None:
            node_importance = node_mask.cpu().numpy()
            # Normalize to [0, 1] for coloring
            node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
        else:
            node_importance = np.ones(mol.GetNumAtoms()) * 0.5
        
        # Create highlight atoms (all atoms for now)
        highlight_atoms = list(range(mol.GetNumAtoms()))
        
        # Create highlight bonds (all bonds for now)
        highlight_bonds = list(range(mol.GetNumBonds()))
        
        # Create atom colors based on importance
        atom_colors = {}
        for i, importance in enumerate(node_importance):
            # Use a colormap: blue (low) to red (high)
            color = plt.cm.RdYlBu_r(importance)[0]
            logger.debug(f"Atom {i} importance: {importance}, color: {color}")
            atom_colors[i] = (color[0], color[1], color[2])
        
        # Create bond colors based on edge importance
        bond_colors = {}
        edge_importance_np = None
        if edge_mask is not None:
            edge_importance_np = edge_mask.cpu().numpy()
            if len(edge_importance_np) > 0:
                edge_importance_np = (edge_importance_np - edge_importance_np.min()) / (edge_importance_np.max() - edge_importance_np.min() + 1e-8)
                
                for i, importance in enumerate(edge_importance_np):
                    if i < mol.GetNumBonds():  # Ensure we don't exceed the number of bonds
                        color = plt.cm.RdYlBu_r(importance)
                        bond_colors[i] = (color[0], color[1], color[2])
            
            # Fill remaining bonds with default color if needed
            for i in range(len(edge_importance_np), mol.GetNumBonds()):
                bond_colors[i] = (0.5, 0.5, 0.5)
        else:
            # Default bond colors
            for i in range(mol.GetNumBonds()):
                bond_colors[i] = (0.5, 0.5, 0.5)
        
        # Generate 2D coordinates
        rdDepictor.Compute2DCoords(mol)
        
        # Create SVG drawer
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
        options = drawer.drawOptions()
        options.padding = 0.0 # Remove padding
        drawer.SetDrawOptions(options)
        
        # Draw molecule with highlights
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=highlight_bonds,
            highlightBondColors=bond_colors
        )
        
        drawer.FinishDrawing()
        
        # Get SVG string
        svg_data = drawer.GetDrawingText()
        
        return {
            'svg_data': svg_data,
            'atom_colors': atom_colors,
            'bond_colors': bond_colors,
            'node_importance': node_importance.tolist(),
            'edge_importance': edge_importance_np.tolist() if edge_mask is not None and edge_importance_np is not None else None
        }
    
    def _get_atom_name(self, mol, atom_idx):
        """Get the atom name/symbol for a given atom index."""
        try:
            if mol is not None and atom_idx < mol.GetNumAtoms():
                atom = mol.GetAtomWithIdx(atom_idx)
                return f"{atom.GetSymbol()}_{atom_idx}"
            return f"Atom_{atom_idx}"
        except:
            return f"Atom_{atom_idx}"
    
    def _create_networkx_graph(self, x, edge_index, node_mask, edge_mask, mol=None):
        """Create a NetworkX graph for additional analysis."""
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes with atom names
            for i in range(x.size(0)):
                importance = float(node_mask[i]) if node_mask is not None else 0.5
                atom_name = self._get_atom_name(mol, i)
                G.add_node(i, importance=importance, atom_name=atom_name)
            
            # Add edges
            edge_importance = edge_mask.cpu().numpy() if edge_mask is not None else np.ones(edge_index.size(1)) * 0.5
            for i in range(edge_index.size(1)):
                src, dst = edge_index[:, i].cpu().numpy()
                importance = float(edge_importance[i])
                G.add_edge(int(src), int(dst), importance=importance)
            
            return G
            
        except Exception as e:
            logger.error(f"Error creating NetworkX graph: {str(e)}")
            return None
    
    def get_visualization_figures(self, explanation_result):
        """
        Generate visualization figures for the explanation results.
        
        Args:
            explanation_result: Dictionary from explain_graph method
            
        Returns:
            Dictionary containing Plotly figures for visualization
        """
        figures = {}
        
        
        # 5. Network visualization
        if explanation_result['nx_graph'] is not None:
            figures['network_plot'] = self._plot_network(explanation_result['nx_graph'], explanation_result.get('mol_obj'))

        # 1. Node importance bar chart
        if explanation_result['top_nodes']:
            figures['node_importance'] = self._plot_node_importance(
                explanation_result['top_nodes'], 
                explanation_result.get('mol_obj')
            )
        
        # 2. Edge importance bar chart
        if explanation_result['top_edges']:
            figures['edge_importance'] = self._plot_edge_importance(
                explanation_result['top_edges'],
                explanation_result.get('mol_obj')
            )
        
        logger.info("Distributions disabled for small drugs")
        # 3. Node importance distribution
        if explanation_result['node_mask'] is not None:
            figures['node_distribution'] = self._plot_node_distribution(explanation_result['node_mask'])
        # 4. Edge importance distribution
        if explanation_result['edge_mask'] is not None:
            figures['edge_distribution'] = self._plot_edge_distribution(explanation_result['edge_mask'])
        
        return figures
    
    def _plot_node_importance(self, top_nodes, mol=None):
        """Create a bar chart of top node importance using matplotlib."""
        if not top_nodes:
            return None

        # Filter out nodes with zero importance
        top_nodes = [node for node in top_nodes if node['importance'] > 0]
        if not top_nodes:
            return None

        # Sort by importance for plotting
        top_nodes = sorted(top_nodes, key=lambda x: x['importance'])

        indices = [node['index'] for node in top_nodes]
        importance = [node['importance'] for node in top_nodes]
        
        if mol is not None:
            labels = [f"{self._get_atom_name(mol, idx)} ({idx})" for idx in indices]
        else:
            labels = [f"Node {idx}" for idx in indices]
        
        fig, ax = get_styled_figure_ax(figsize=(10, len(labels) * 0.5), aspect=None, grid=True)
        ax.barh(np.arange(len(labels)), importance, color='lightblue', align='center')
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Atom (Node Index)")
        ax.set_title("Top Node Importance (GNN Explainer)")
        fig.tight_layout()
        
        return fig
    
    def _plot_edge_importance(self, top_edges, mol=None):
        """Create a bar chart of top edge importance using matplotlib."""
        if not top_edges:
            return None

        # Filter out edges with zero importance
        top_edges = [edge for edge in top_edges if edge['importance'] > 0]
        if not top_edges:
            return None

        # Sort by importance for plotting
        top_edges = sorted(top_edges, key=lambda x: x['importance'])
        
        edge_labels = [f"{edge['source']}-{edge['target']}" for edge in top_edges]
        importance = [edge['importance'] for edge in top_edges]
        
        fig, ax = get_styled_figure_ax(figsize=(10, len(edge_labels) * 0.5), aspect=None, grid=True)
        ax.barh(np.arange(len(edge_labels)), importance, color='lightcoral', align='center')
        ax.set_yticks(np.arange(len(edge_labels)))
        ax.set_yticklabels(edge_labels)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Bond (Atom-Atom)")
        ax.set_title("Top Edge Importance (GNN Explainer)")
        fig.tight_layout()
        
        return fig
    
    def _plot_node_distribution(self, node_mask):
        """Create a histogram of node importance distribution using matplotlib."""
        node_importance = node_mask.cpu().numpy()
        
        fig, ax = get_styled_figure_ax(figsize=(8, 6), aspect=None, grid=True)
        ax.hist(node_importance, bins=30, color='lightgreen', alpha=0.7)
        ax.set_title("Node Importance Distribution")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Count")
        fig.tight_layout()
        
        return fig
    
    def _plot_edge_distribution(self, edge_mask):
        """Create a histogram of edge importance distribution using matplotlib."""
        edge_importance = edge_mask.cpu().numpy()

        fig, ax = get_styled_figure_ax(figsize=(8, 6), aspect=None, grid=True)
        ax.hist(edge_importance, bins=30, color='lightpink', alpha=0.7)
        ax.set_title("Edge Importance Distribution")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Count")
        fig.tight_layout()
        
        return fig
    
    def _plot_network(self, nx_graph, mol=None):
        """Create a network visualization of the molecular graph."""
        try:
            # Determine layout
            pos = None
            
            # 1. Try to use molecule coordinates if available (matches the 2D structure)
            if mol is not None:
                try:
                    if mol.GetNumConformers() > 0:
                        pos = {}
                        conf = mol.GetConformer()
                        for node in nx_graph.nodes():
                            # Node key is the atom index
                            p = conf.GetAtomPosition(int(node))
                            # RDKit coordinates: x, y. (z is usually 0 for 2D)
                            pos[node] = (p.x, p.y)
                except Exception as e:
                    logger.warning(f"Could not use molecule coordinates for layout: {e}")
                    pos = None

            # 2. Fallback layouts if no molecule coordinates
            if pos is None:
                # Try different layouts for better planar visualization
                try:
                    # Try planar layout first
                    pos = nx.planar_layout(nx_graph)
                except:
                    try:
                        # Try circular layout
                        pos = nx.circular_layout(nx_graph)
                    except:
                        # Fall back to spring layout with better parameters
                        pos = nx.spring_layout(nx_graph, k=2, iterations=100, seed=42)
            
            # Extract node and edge data
            node_x = [pos[node][0] for node in nx_graph.nodes()]
            node_y = [pos[node][1] for node in nx_graph.nodes()]
            node_importance = [nx_graph.nodes[node]['importance'] for node in nx_graph.nodes()]
            
            if mol is not None:
                node_names = [self._get_atom_name(mol, int(node)) for node in nx_graph.nodes()]
            else:
                node_names = [nx_graph.nodes[node].get('atom_name', f'Node {node}') for node in nx_graph.nodes()]
            
            # Create edge traces with importance-based coloring
            edge_x = []
            edge_y = []
            edge_importance = []
            edge_colors = []
            
            for edge in nx_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                importance = nx_graph.edges[edge]['importance']
                edge_importance.append(importance)
                # Color edges based on importance
                edge_colors.append(importance)
            
            # Create the plot
            fig = go.Figure()
            
            # Add edges with importance-based coloring
            if edge_colors:
                # Normalize edge colors
                edge_colors_norm = [(c - min(edge_colors)) / (max(edge_colors) - min(edge_colors) + 1e-8) 
                                  for c in edge_colors]
                
                # Create edge traces with different colors
                for i, (x0, y0, x1, y1) in enumerate(zip(edge_x[::3], edge_y[::3], edge_x[1::3], edge_y[1::3])):
                    color_intensity = edge_colors_norm[i]
                    color = f'rgba({int(255 * color_intensity)}, {int(100 * (1-color_intensity))}, {int(100 * (1-color_intensity))}, 0.8)'
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        line=dict(width=3, color=color),
                        hoverinfo='none',
                        mode='lines',
                        showlegend=False
                    ))
            else:
                # Default edge color
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='gray'),
                    hoverinfo='none',
                    mode='lines',
                    name='Bonds'
                ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=25,
                    color=node_importance,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Node Importance"),
                    line=dict(width=3, color='black')
                ),
                text=node_names,
                textposition="middle center",
                textfont=dict(size=12, color='black'),
                hovertemplate="Atom: %{text}<br>Importance: %{marker.color:.4f}<extra></extra>",
                name='Atoms'
            ))
            
            fig.update_layout(
                title="Molecular Graph Network Visualization",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Node color and size represent importance. Edge color represents bond importance.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating network plot: {str(e)}")
            return None


if __name__ == '__main__':
    # Test the GNN explainer
    model_config = {
        'block_num': 3,
        'vocab_protein_size': 25 + 1,
        'embedding_size': 128,
        'filter_num': 32,
        'out_dim': 1
    }
    
    device = 'cpu'
    model = MGraphDTA(**model_config).to(device)
    model_path = os.path.join(os.getcwd(), "models", "epoch-78, loss-0.1761, cindex-0.8863, test_loss-0.2366.pt")
    model_dict = torch.load(model_path, weights_only=True, map_location=device)
    
    model.load_state_dict(model_dict)
    model.eval()
    
    # Initialize GNN explainer
    gnn_explainer = GNNExplainerWrapper(model, dataset_name='full_toxcast', device=device)
    

     

    # Test explanation
    print("Testing GNN explanation...")
    explanation_result = gnn_explainer.explain_graph(*gnn_explainer.get_data(0), data_set_idx=0)
    print(f"GNN explanation completed. Prediction: {explanation_result['original_prediction']:.4f}")
    print(f"Top nodes: {len(explanation_result['top_nodes'])}")
    print(f"Top edges: {len(explanation_result['top_edges'])}")
    
    # Test visualization
    print("Generating visualization figures...")
    figures = gnn_explainer.get_visualization_figures(explanation_result)
    print(f"Generated {len(figures)} visualization figures")
    
    # Save figures
    for name, fig in figures.items():
        if fig is not None:
            fig.write_html(f'gnn_{name}_explanation.html')
            print(f"Saved {name} figure as 'gnn_{name}_explanation.html'")
