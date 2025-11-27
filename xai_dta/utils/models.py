import os 
import torch 

from MGraphDTA.regression.model import MGraphDTA

def load_model(root:str='', device='cpu', model_name='epoch-134, loss-0.1411, cindex-0.9066, test_loss-0.1752.pt') -> MGraphDTA:
    """Load model and return in eval mode"""
    model_config = {
        'block_num': 3,
        'vocab_protein_size': 25 + 1,
        'embedding_size': 128,
        'filter_num': 32,
        'out_dim': 1
    }
    
    if root == '':
        root = os.getcwd()
    model = MGraphDTA(**model_config).to(device)
    model_path = os.path.join(root, "models", model_name)
    model_dict = torch.load(model_path, weights_only=True, map_location=device)
    
    model.load_state_dict(model_dict)
    model.eval()

    return model

def load_untrained_model(device='cpu') -> MGraphDTA:
    model_config = {
        'block_num': 3,
        'vocab_protein_size': 25 + 1,
        'embedding_size': 128,
        'filter_num': 32,
        'out_dim': 1
    }
    model = MGraphDTA(**model_config).to(device)
    model.eval()
    return model

