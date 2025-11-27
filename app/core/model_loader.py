import streamlit as st
import os
import torch
from MGraphDTA.regression.model import MGraphDTA
from xai.explainer import Explainer
from xai.explainer_gnn import GNNExplainerWrapper
from xai_dta.config import PROJ_ROOT

@st.cache_resource
def load_model(device='cpu', saved_model_path:str='', dataset_name='kiba') -> Explainer:
    """Load model once and cache it"""
    model_config = {
        'block_num': 3,
        'vocab_protein_size': 25 + 1,
        'embedding_size': 128,
        'filter_num': 32,
        'out_dim': 1
    }
    
    model = MGraphDTA(**model_config).to(device)
    # model_path = os.path.join(os.getcwd(), "models", "epoch-10, loss-0.5653, cindex-0.7677, test_loss-0.5274.pt")
    if saved_model_path:
        model_path = saved_model_path
    else:
        # model_path = os.path.join(os.getcwd(), "models", dataset_name, "epoch-2159, loss-0.0179, cindex-0.9590, test_loss-0.1286.pt")
        model_path = os.path.join(PROJ_ROOT, "models", dataset_name, 
                                  # "fold-0, repeat-0, epoch-2320, train_loss-0.0028, train_cindex-0.9946, val_loss-0.1456, val_cindex-0.8923, val_r2-0.7678.pt"
                                  # "fold-1, repeat-1, epoch-2026, train_loss-0.0033, train_cindex-0.9936, val_loss-0.1394, val_cindex-0.8940, val_r2-0.7582.pt"
                                  "epoch-2578, loss-0.0122, cindex-0.9741, test_loss-0.1265.pt"
                                  )
    model_dict = torch.load(model_path, weights_only=True, map_location=device)
    
    model.load_state_dict(model_dict)
    model.eval()
    
    # explainer = Explainer(model, dataset_name='davis', device=device)
    explainer = Explainer(model, dataset_name=dataset_name, device=device)

    return explainer

@st.cache_resource
def load_gnn_explainer(_model, device='cpu', dataset_name='kiba') -> GNNExplainerWrapper:
    """Load GNN explainer once and cache it"""
    
    gnn_explainer = GNNExplainerWrapper(_model, dataset_name=dataset_name, device=device)
    
    return gnn_explainer
