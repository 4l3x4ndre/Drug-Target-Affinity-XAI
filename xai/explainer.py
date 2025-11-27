import numpy as np
import os 
import pandas as pd 
import torch
import copy
from torch_geometric.data import Batch
import shap
from loguru import logger

from MGraphDTA.visualization.visualization_mgnn import GradAAM
from MGraphDTA.regression.preprocessing import GNNDataset
from MGraphDTA.regression.model import MGraphDTA
from xai.xai_model import FullModelForGradCAM
from tqdm import tqdm
from GradCam.pytorch_grad_cam import GradCAM
from MGraphDTA.visualization.preprocessing import VOCAB_PROTEIN

from lime import lime_tabular

INT2VOCAB = {str(v): k for k, v in VOCAB_PROTEIN.items()}
INT2VOCAB['0'] = '-'

class RegressionOutputTarget:
    def __call__(self, model_output):
        return model_output.squeeze()

def reshape_transform_1d(activations):
    if activations.dim() == 4:
        return activations
    if activations.dim() == 3:
        return activations.unsqueeze(2)
    if activations.dim() == 2:
        return activations.unsqueeze(1).unsqueeze(2)
    raise ValueError(f"Unexpected activations dim: {activations.dim()}")


class ShapModelWrapper:
    def __init__(self, model, device, ligand_data):
        self.model = model
        self.device = device
        self.model.eval()
        self.base_ligand_data = copy.deepcopy(ligand_data)
        delattr(self.base_ligand_data, 'target')

    def __call__(self, protein_sequences):
        """
        Forward pass for SHAP explainer.
        protein_sequences: numpy array of shape (n_samples, sequence_length)
        """
        if isinstance(protein_sequences, np.ndarray):
            if protein_sequences.ndim == 1:
                protein_sequences = protein_sequences.reshape(1, -1)
        else:
            protein_sequences = np.array(protein_sequences)
            if protein_sequences.ndim == 1:
                protein_sequences = protein_sequences.reshape(1, -1)
        
        predictions = []
        batch_size = 32  # Process in batches to manage memory
        
        for i in tqdm(range(0, protein_sequences.shape[0], batch_size), desc="SHAP batches"):
            batch_sequences = protein_sequences[i:i+batch_size]
            current_batch_size = batch_sequences.shape[0]
            
            # Create batch of data objects
            data_list = []
            for j in range(current_batch_size):
                ligand_copy = self.base_ligand_data.clone()
                ligand_copy.target = torch.LongTensor(batch_sequences[j]).unsqueeze(0)
                data_list.append(ligand_copy)
            
            # Create batch and get predictions
            batch = Batch.from_data_list(data_list).to(self.device)
            
            with torch.no_grad():
                out = self.model(batch).cpu().numpy()
                predictions.append(out)
        
        return np.vstack(predictions)


class Explainer():
    def __init__(self, model, dataset_name='davis', device='cpu', path_prefix=''):
        self.model = model
        self.device = device


        if len(path_prefix)==0:
            fpath = os.path.join("MGraphDTA/regression/data", dataset_name)
        else:
            fpath = os.path.join(path_prefix, "MGraphDTA/regression/data", dataset_name)
        test_df = pd.read_csv(os.path.join(fpath, 'raw', 'data_test.csv'))
        self.dataset = GNNDataset(fpath, train=False, transform_unique=True, inference_mode=True)
        logger.info(f"Loaded dataset from {fpath}")
        self.dataset_name = dataset_name
        self.df = test_df

        if dataset_name == 'davis' or dataset_name == 'kiba':
            self.smile_list = test_df['compound_iso_smiles'].values
            self.target_list = test_df['target_sequence'].values
            self.labels = test_df['affinity'].values
        elif dataset_name == 'full_toxcast':
            self.smile_list = test_df['smiles'].values
            self.target_list = test_df['sequence'].values
            self.labels = test_df['label'].values

        # Manually process the first 500 sequences for background data
        # This avoids loading the heavy .pt files in the dataset
        background_sequences = []
        for i in tqdm(range(min(len(self.target_list), 500)), desc="Preparing background sequences for LIME"):
            seq = self.target_list[i]
            # Simple processing matching GNNDataset logic
            target = []
            for s in seq:
                if s in VOCAB_PROTEIN:
                    target.append(VOCAB_PROTEIN[s])
                else:
                    target.append(0)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            background_sequences.append(target)
            
        self.protein_training_data = np.array(background_sequences)

    def get_data(self, idx):
        logger.info(f"Fetching data at index: {idx} out of {len(self.smile_list)}")
        return self.smile_list[idx], self.target_list[idx]
    def get_smile(self, idx):
        return self.smile_list[idx]
    def get_target(self, idx):
        return self.target_list[idx]
    def get_dataset_idx(self, smile, target):
        for i in range(len(self.smile_list)):
            if self.smile_list[i]==smile and self.target_list[i]==target:
                return i
        return -1
    def __len__(self):
        return len(self.smile_list)

    def set_mode(self, mode='eval'):
        if mode == 'eval':
            self.model.eval()
        elif mode == 'train':
            self.model.train()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def explain(self, smiles_input, protein_input, data_set_idx=None):
        data, gm = self.dataset.transform_unique(smiles_input, protein_input)
        g, mol = gm
        data = data.to(self.device)

        data_copy = data.clone()

        # Setup GradCAM
        target_layers = [
            self.model.protein_encoder.block_list[0].inc.conv_layer0.inc[0], # Layer 0 <==
            self.model.protein_encoder.block_list[1].inc.conv_layer0.inc[0], # Layer 1
            self.model.protein_encoder.block_list[1].inc.conv_layer1.inc[0], # Layer 2 <==
            self.model.protein_encoder.block_list[2].inc.conv_layer0.inc[0], # Layer 3
            self.model.protein_encoder.block_list[2].inc.conv_layer1.inc[0], # Layer 4
            self.model.protein_encoder.block_list[2].inc.conv_layer2.inc[0]  # Layer 5 <==
        ]
        layer_names = ['Layer 1.1', 'Layer 2.1', 'Layer 2.2', 'Layer 3.1', 'Layer 3.2', 'Layer 3.3']

        # Setup GradAAM for ligand
        gradcam = GradAAM(self.model, module=self.model.ligand_encoder.features.transition3)

        # Process protein CAMs
        protein_sequence = data.target.to(self.device)
        input_tensor_4d = protein_sequence.unsqueeze(1).unsqueeze(2)

        per_layer_cams = []
        for tl in target_layers:
            wrapped_model = FullModelForGradCAM(self.model, data).to(self.device)
            cam = GradCAM(model=wrapped_model, target_layers=[tl], 
                        reshape_transform=reshape_transform_1d)
            grayscale_cam = cam(input_tensor=input_tensor_4d, 
                              targets=[RegressionOutputTarget()])
            cam_map = grayscale_cam[0]
            if cam_map.ndim == 2 and cam_map.shape[0] == 1:
                cam_1d = cam_map.squeeze(0)
            else:
                cam_1d = cam_map.flatten()
            per_layer_cams.append(cam_1d)

        # Get predictions and ligand visualization
        prediction, protein_x, ligand_x, atom_att = gradcam.forward_features(data)

        output= {
            'prediction':prediction.item(), 
            'per_layer_cams':per_layer_cams, 
            'atom_att':atom_att, 
            'graph_obj':g, 
            'mol_obj':mol,
            'layer_names':layer_names,
            'protein_x':protein_x,
            'protein_sequence':protein_sequence,
            'data_obj': data_copy
        }
        if data_set_idx is not None:
            output['true_affinity'] = self.labels[data_set_idx]
            if self.dataset_name == 'full_toxcast':
                output['uniprot_id'] = self.df['Uniprot_ID'].values[data_set_idx]
        return output


    def explain_lime(self, smiles_input, protein_input):
        print(protein_input)
        data, gm = self.dataset.transform_unique(smiles_input, protein_input)
        g, mol = gm
        data = data.to(self.device)


        # --- Find the true length of the protein before padding ---
        original_sequence_tensor = data.target.cpu().numpy().flatten()
        # Find the index of the last non-zero element
        true_indices = np.where(original_sequence_tensor != 0)[0]
        true_length = true_indices[-1] + 1 if len(true_indices) > 0 else 0
        print(f"Detected true protein length: {true_length}")
        # -----------------------------------------------------------------

        def model_predict(protein_seq_numpy):
            # --- Zero-out the padding on perturbed samples ---
            # This ensures the model never sees perturbations in the padded area.
            protein_seq_numpy[:, true_length:] = 0
            # --------------------------------------------------------

            # Convert LIME's numpy array of perturbed sequences to a tensor
            protein_seq_tensor = torch.LongTensor(protein_seq_numpy).to(self.device)
            outputs_list = []
            batch_size = 32  # Process in batches to manage memory

            # We need to pair the original ligand graph with each perturbed protein.
            base_ligand_data = copy.deepcopy(data)
            delattr(base_ligand_data, 'target') # Remove the original protein to avoid confusion

            for i in range(0, protein_seq_tensor.size(0), batch_size):
                batch_protein_sequences = protein_seq_tensor[i:i+batch_size]
                current_batch_size = batch_protein_sequences.size(0)

                data_list = []
                for j in range(current_batch_size):
                    ligand_copy = base_ligand_data.clone()
                    ligand_copy.target = batch_protein_sequences[j].unsqueeze(0)
                    data_list.append(ligand_copy)
                
                batch = Batch.from_data_list(data_list).to(self.device)

                with torch.no_grad():
                    out = self.model(batch).cpu().numpy()
                    outputs_list.append(out)

            return np.vstack(outputs_list)

        # --- LIME Categorical Feature Setup ---
        # Create feature names like 'pos_0', 'pos_1', etc.
        feature_names = [f'pos_{i}' for i in range(data.target.size(1))]

        # Define which features are categorical (all of them)
        categorical_features = list(range(true_length))

        # Create the mapping from integer value to amino acid string
        INT2VOCAB_LIME = {v: k for k, v in VOCAB_PROTEIN.items()}
        INT2VOCAB_LIME[0] = '-' 
        vocab_size = 26 
        categorical_names_list = [INT2VOCAB_LIME.get(i, '?') for i in range(vocab_size)]
        
        # LIME expects a dict mapping feature index to a list of names
        categorical_names = {i: categorical_names_list for i in categorical_features}

        # Create a LIME explainer for tabular data
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.protein_training_data,
            mode='regression',
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_names=categorical_names,
            discretize_continuous=False # Treat features as categorical
        )

        # Generate the explanation
        lime_explanation = lime_explainer.explain_instance(
            data_row=data.target.cpu().numpy().flatten(),
            predict_fn=model_predict,
            num_samples=5000,
            num_features=true_length # Explain all non-padding features
        )

        explanation_list = lime_explanation.as_list()
        explanation_df = pd.DataFrame(explanation_list, columns=['feature', 'weight'])
        
        # Extract position number from feature string (e.g., "pos_42 = A" -> 42)
        # Using a regular expression to find numbers after 'pos_'
        position_numbers = explanation_df['feature'].str.extract(r'pos_(\d+)').astype(int)
        
        # Add a temporary column for sorting
        explanation_df['position_order'] = position_numbers
        
        # Sort by position_order, then drop the temporary column
        explanation_df = explanation_df.sort_values(by='position_order', ascending=False).drop(columns=['position_order'])
        explanation_df['color'] = explanation_df['weight'].apply(lambda x: 'green' if x > 0 else 'red')

        return {
            'explanation_df': explanation_df,
            'lime_explanation': lime_explanation,
        }

    def explain_shap(self, smiles_input, protein_input, background_size=100, num_samples=4):
        """
        Explain predictions using SHAP KernelExplainer for PyTorch models.
        
        Args:
            smiles_input: SMILES string for the ligand
            protein_input: Protein sequence string
            background_size: Number of background samples for SHAP explainer
            num_samples: Number of test samples to explain
            
        Returns:
            Dictionary containing SHAP values and visualization data
        """
        data, _ = self.dataset.transform_unique(smiles_input, protein_input)
        data = data.to(self.device)
        
        # Get the protein sequence tensor
        protein_sequence = data.target.cpu().numpy()
        
        # Create background dataset from training data
        background_indices = np.random.choice(
            len(self.protein_training_data), 
            min(background_size, len(self.protein_training_data)), 
            replace=False
        )
        background = self.protein_training_data[background_indices]
        
        # Convert background to torch tensors
        background_tensors = []
        for bg_seq in background:
            # Create a copy of the original data with the background protein sequence
            bg_data = copy.deepcopy(data)
            bg_data.target = torch.LongTensor(bg_seq).unsqueeze(0).to(self.device)
            background_tensors.append(bg_data)
        
        # Create model wrapper
        model_wrapper = ShapModelWrapper(self.model, self.device, data)
        
        # Initialize SHAP KernelExplainer for PyTorch
        explainer = shap.KernelExplainer(model_wrapper, background)
        
        # Generate SHAP values for the test sample
        test_sample = protein_sequence.reshape(1, -1)
        shap_values = explainer.shap_values(test_sample, nsamples=100)  # TODO
        
        # Get the base value (expected value)
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value.item()

        # Get the original prediction
        original_prediction = model_wrapper(protein_sequence)[0][0]
        
        # Create visualization data
        # For protein sequences, 2D representation for image plot
        # Reshape to (1, 1, sequence_length) for image plot compatibility
        shap_values_reshaped = shap_values.reshape(1, 1, -1)
        test_sample_reshaped = test_sample.reshape(1, 1, -1)
        
        # Create amino acid labels for the sequence
        protein_sequence_flat = protein_sequence.flatten()
        amino_acid_labels = []
        for i, aa_id in enumerate(protein_sequence_flat):
            if aa_id == 0:  # Padding
                amino_acid_labels.append('-')
            else:
                amino_acid_labels.append(INT2VOCAB.get(str(aa_id), '?'))
        
        return {
            'shap_values': shap_values,
            'shap_values_reshaped': shap_values_reshaped,
            'test_sample': test_sample,
            'test_sample_reshaped': test_sample_reshaped,
            'background': background,
            'original_prediction': original_prediction,
            'amino_acid_labels': amino_acid_labels,
            'protein_sequence': protein_sequence,
            'base_value': base_value,
            'method': 'KernelExplainer'
        }


if __name__=='__main__':
    model_config = {
        'block_num': 3,
        'vocab_protein_size': 25 + 1,
        'embedding_size': 128,
        'filter_num': 32,
        'out_dim': 1
    }
    
    device = 'cpu'
    model = MGraphDTA(**model_config).to(device)
    model_path = os.path.join(os.getcwd(), "models", "epoch-10, loss-0.5653, cindex-0.7677, test_loss-0.5274.pt")
    model_dict = torch.load(model_path, weights_only=True, map_location=device)
    
    model.load_state_dict(model_dict)
    model.eval()
    
    explainer = Explainer(model, dataset_name='full_toxcast', device=device)

    # Test SHAP explanation (Kernel)
    print("Testing SHAP Kernel explanation...")
    shap_results = explainer.explain_shap(*explainer.get_data(0))
    print(f"SHAP Kernel explanation completed. Prediction: {shap_results['original_prediction']:.4f}")
    print(f"SHAP values shape: {shap_results['shap_values'].shape}")
    print(f"Sequence length: {len(shap_results['amino_acid_labels'])}")
    
    # Test visualization with enhanced Plotly functions
    from xai.visualization.shap_visual import plot_shap_waterfall, create_shap_summary_cards
    
    # Test waterfall plots
    fig_waterfall_kernel = plot_shap_waterfall(shap_results)
    fig_waterfall_kernel.write_html('shap_waterfall_kernel.html')
    print("SHAP Kernel waterfall plot saved as 'shap_waterfall_kernel.html'")
    
    # Test summary cards
    summary_kernel = create_shap_summary_cards(shap_results)
    
    print("\n=== SHAP Summary Comparison ===")
    print(f"Kernel Method - Prediction: {summary_kernel['prediction']:.4f}, Mean |SHAP|: {summary_kernel['mean_absolute_contribution']:.4f}")
    print(f"Top positive position (Kernel): {summary_kernel['top_positive_position']} ({summary_kernel['top_positive_aa']})")
    
