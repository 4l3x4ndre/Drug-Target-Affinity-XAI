import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
import argparse
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

from MGraphDTA.visualization.dataset import *
from MGraphDTA.visualization.model import MGraphDTA
from MGraphDTA.visualization.utils import *
from MGraphDTA.visualization.visualization_mgnn import GradAAM, clourMol
from MGraphDTA.visualization.preprocessing import VOCAB_PROTEIN
from xai.xai_model import FullModelForGradCAM

from GradCam.pytorch_grad_cam import GradCAM

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm


INT2VOCAB = {str(v): k for k, v in VOCAB_PROTEIN.items()}
INT2VOCAB['0'] = '-' # padding

class RegressionOutputTarget:
    def __call__(self, model_output):
        # model_output has shape [batch_size, 1]
        # CAM expects a tensor of shape [batch_size]
        return model_output.squeeze()

def reshape_transform_1d(activations):
    """
    Convert activations to shape [B, C, H, W] for pytorch-grad-cam.

    activations might be:
        - [B, C, L] -> return [B, C, 1, L]
        - [B, L]    -> return [B, 1, 1, L]
        - already 4D -> return as-is
    """
    if activations.dim() == 4:
        return activations
    if activations.dim() == 3:
        return activations.unsqueeze(2)  # [B, C, 1, L]
    if activations.dim() == 2:
        return activations.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    raise ValueError(f"Unexpected activations dim: {activations.dim()}")



def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='full_toxcast')
    parser.add_argument('--dataroot', required=True, help='Relative path to data folder eg. MGraphDTA/visualization/data/')
    parser.add_argument('--device', required=True, help='cpu or cuda:0')
    args = parser.parse_args()

    params = dict(
        data_root=args.dataroot,
        save_dir="save",
        dataset=args.dataset,
        device=args.device
    )

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)
    device = params.get("device")
    
    test_df = pd.read_csv(os.path.join(fpath, 'raw', 'data_test.csv'))
    test_set = GNNDataset(fpath, train=False)
    smile_list = list(test_df['smiles'].unique())

    
    # model configuration
    model_config = {
        'block_num': 3,
        'vocab_protein_size': 25 + 1,
        'embedding_size': 128,
        'filter_num': 32,
        'out_dim': 1
    }


    # --- Version for fullmodel
    model = MGraphDTA(**model_config).to(device)
    model_dict = torch.load('MGraphDTA/visualization/pretrained_model/epoch-173, loss-0.1468, cindex-0.8998, test_loss-0.1750.pt', weights_only=True)
    for key, val in model_dict.copy().items():
        if 'lin_l' in key:
            new_key = key.replace('lin_l', 'lin_rel')
            model_dict[new_key] = model_dict.pop(key)
        elif 'lin_r' in key:
            new_key = key.replace('lin_r', 'lin_root')
            model_dict[new_key] = model_dict.pop(key)
    model.load_state_dict(model_dict)
    model.train()
    gradcam = GradAAM(model, module=model.ligand_encoder.features.transition3)

    bottom = cm.get_cmap('Blues_r', 256)
    top = cm.get_cmap('Oranges', 256)
    newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    # Choose the target layer inside your protein encoder (must be a conv / feature map)
    print(model)
    target_layers = [
        model.protein_encoder.block_list[0].inc.conv_layer0.inc[0],
        model.protein_encoder.block_list[1].inc.conv_layer1.inc[0],
        model.protein_encoder.block_list[2].inc.conv_layer2.inc[0]
    ]
    layer_names = ['block_list[0].inc.conv_layer0', 'block_list[1].inc.conv_layer0', 'block_list[2].inc.conv_layer0']
    # ---

    
    progress_bar = tqdm(total=len(smile_list))
    for idx in range(len(test_set)):
        smile = test_df.iloc[idx]['smiles']

        if len(smile_list) == 0:
            break
        if smile in smile_list:
            smile_list.remove(smile)
        else:
            continue

        data = Batch.from_data_list([test_set[idx]])
        data = data.to(device)
        # Build input tensor for GradCAM: [B,1,1,L]:
        protein_sequence = data.target.to(device)   # data from DataLoader
        input_tensor_4d = protein_sequence.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]

        ligand_template = test_set[idx] 
        

        per_layer_cams = []
        for tl_idx, target_l in enumerate(target_layers):
            print(tl_idx, target_l)
            tls = [target_l]

            wrapped_model = FullModelForGradCAM(model, ligand_template).to(device)
            cam = GradCAM(model=wrapped_model, target_layers=tls, reshape_transform=reshape_transform_1d)

            grayscale_cam = cam(input_tensor=input_tensor_4d, targets=[RegressionOutputTarget()])
            # grayscale_cam shape typically [B, H, W] -> H==1, W==L

            # get 1-D cam: flatten width dimension
            cam_map = grayscale_cam[0]  # first batch
            if cam_map.ndim == 2 and cam_map.shape[0] == 1:
                cam_1d = cam_map.squeeze(0)   # shape (L,)
            else:
                cam_1d = cam_map.flatten()
            # cam_1d.shape = (1200,)
            per_layer_cams.append(cam_1d)


        # Get model outputs and intermediate features. 
        prediction, protein_x, ligand_x, atom_att = gradcam.forward_features(data)
        mol = Chem.MolFromSmiles(smile)
        atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) for idx in range(len(atom_att))])
        radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
        ligand_img = clourMol(mol,highlightAtoms_p=range(len(atom_att)), highlightAtomColors_p=atom_color, radii=radii)

        prediction = prediction[0] # becomes a float
        y_label = data.y[0]
        
        visualize_protein_cam(per_layer_cams, layer_names,
                              protein_x.detach().numpy(),
                              ligand_img,
                              prediction,
                              y_label,
                              protein_sequence[0],
                              VOCAB_PROTEIN, 
                              f'results/{idx:05d}.png')

    return

    
def visualize_protein_cam(grayscale_cam_list,
                          layer_names,
                          encoder_output_features,
                          gradaam_output,
                          prediction:float,
                          y_label:float|int,
                          protein_sequence, amino_acid_vocab=None,
                          save_path='results/protein_cam.png',
                          max_per_row=60):
    """
    Visualize one or more CAM results for a protein sequence.

    Args:
        grayscale_cam_list: single CAM (ndarray/torch tensor) or a list/ndarray of CAMs.
            Each CAM may be 1D (L,), 2D (H,W) or 3D. We reduce it to a 1D map per CAM.
        layer_names: list of str representing the name of each layer for which cam is 
            called.
        encoder_output_features: numpy array of encoder's output (1, 96)
        gradaam_output: image of Grad-AAM
        prediction: MGraphDTA output
        protein_sequence: Original protein sequence tensor (1D)
        amino_acid_vocab: Optional mapping {token_id: 'amino_acid_letter'}
        save_path: Where to save the visualization
        max_per_row: how many sequence tokens to show per row
    """
    print("Start")

    # Normalize input to a Python list of numpy arrays
    if isinstance(grayscale_cam_list, (np.ndarray, torch.Tensor)) and not isinstance(grayscale_cam_list, list):
        # If ndarray and 3D: treat as list of CAMs along axis 0
        arr = grayscale_cam_list
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3:
            cams = [arr[i] for i in range(arr.shape[0])]
        else:
            cams = [arr]
    elif isinstance(grayscale_cam_list, list):
        cams = []
        for c in grayscale_cam_list:
            if isinstance(c, torch.Tensor):
                cams.append(c.cpu().numpy())
            else:
                cams.append(np.asarray(c))
    else:
        # fallback
        cams = [np.asarray(grayscale_cam_list)]

    # Convert sequence to numpy and letters
    if isinstance(protein_sequence, torch.Tensor):
        seq_np = protein_sequence.detach().cpu().flatten().numpy()
    else:
        seq_np = np.asarray(protein_sequence).flatten()

    seq_len = len(seq_np)

    print(f"Got {len(cams)} CAM(s). Sequence length = {seq_len}")

    # Helper: convert a CAM (various shapes) to 1D numpy array
    def cam_to_1d(cam):
        cam = np.asarray(cam)
        if cam.ndim == 1:
            cam1 = cam
        elif cam.ndim == 2:
            # flatten spatial map
            cam1 = cam.flatten()
        elif cam.ndim == 3:
            # average over first dimension (channels / layers), then flatten
            cam1 = cam.mean(axis=0).flatten()
        else:
            cam1 = cam.flatten()
        return cam1

    # Prepare seq letters
    if amino_acid_vocab:
        seq_letters_full = [INT2VOCAB.get(str(int(k)), '-') for k in seq_np]
    else:
        seq_letters_full = [f'{int(k)}' for k in seq_np]

    # For each CAM: trim / pad to seq_len and normalize
    cam_1d_list = []
    for idx, cam in enumerate(cams):
        cam1 = cam_to_1d(cam)
        # Trim or pad
        if cam1.size > seq_len:
            cam1 = cam1[:seq_len]
        elif cam1.size < seq_len:
            cam1 = np.pad(cam1, (0, seq_len - cam1.size), mode='constant')
        cam_1d_list.append(cam1)

    # Layout: 3 rows x n_cols
    n_cols = len(cam_1d_list)
    n_rows = int(np.ceil(seq_len / max_per_row))
    fig_height = max(4, 6 + .5 * n_rows)    #
    fig_width = max(10, 7 * n_cols)         #

    # fig, axes = plt.subplots(4, n_cols, figsize=(fig_width, fig_height) , gridspec_kw={'height_ratios': [0.25, 0.1, 1, .1]})
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(5, n_cols, height_ratios=[0.25, 0.1, 1, .2, .4])

    for col_idx, cam1_full in enumerate(cam_1d_list):
        

        # Cut out the last portion of weights that has zero attention weights.
        threshold = 1e-4                                 # Define threshold for "non-infinitesimal" values
        indices = np.flatnonzero(cam1_full > threshold)  # Find last index above threshold (if any) 
        if indices.size > 0:
            last_index = min(len(cam1_full)-1, indices[-1]+10)
            last_index = max(                           # Make sure to keep full protein sequence
                len("".join(seq_letters_full[:last_index+1]).strip('-'))+10, 
                last_index
            )
        else:
            last_index = min(                           # Remove padding
                len("".join(seq_letters_full).strip('-'))+10, 
               len(cam1_full)-1 
            )
        cam1 = cam1_full[:last_index + 1]
        positions = np.arange(last_index+1)
        seq_letters = seq_letters_full[:last_index+1]
        print('Cam1.min()=', cam1.min(), 'last:', cam1[-1])


        ax1 = fig.add_subplot(gs[0, col_idx]) 
        ax2 = fig.add_subplot(gs[1, col_idx]) 
        ax3 = fig.add_subplot(gs[2, col_idx]) 

        # Plot 1: Line plot
        ax1.plot(positions, cam1, '-', linewidth=1.5, alpha=0.8)
        ax1.fill_between(positions, cam1, alpha=0.25)
        ax1.set_title(f'{layer_names[col_idx]}\nAttention Along Sequence')
        ax1.set_xlabel('Sequence Position')
        ax1.set_ylabel('Attention Weight')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.25)

        # Plot 2: Heatmap
        cam_2d = cam1.reshape(1, -1)
        im2 = ax2.imshow(cam_2d, cmap='Reds', aspect='auto')
        ax2.set_title('Attention Heatmap')
        ax2.set_xlabel('Sequence Position')
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, orientation='vertical', pad=0.02)

        # Plot 3: Sequence with colored background
        cmap_seq = LinearSegmentedColormap.from_list('custom', ['white', 'red'], N=100)
        norm_weights = (cam1 - cam1.min()) / (cam1.max() - cam1.min() + 1e-8)

        ax3.set_xlim(-0.5, max_per_row - 0.5)
        ax3.set_ylim(-0.5, n_rows - 0.5)

        for idx_token, (letter, weight) in enumerate(zip(seq_letters, norm_weights)):
            row = idx_token // max_per_row
            col = idx_token % max_per_row
            ax3.add_patch(plt.Rectangle((col-0.4, n_rows - row - 1 - 0.4), 0.8, 0.8,
                                        facecolor=cmap_seq(weight), alpha=0.8))
            display_letter = letter
            if display_letter == 'PAD':
                display_letter = '-'
            elif display_letter == 'START':
                display_letter = '>'
            elif display_letter == 'END':
                display_letter = '||'
            ax3.text(col, n_rows - row - 1, display_letter,
                     ha='center', va='center', fontsize=8, fontweight='bold')

        ax3.set_title(f'Protein Sequence with Attention Coloring\n(colors normalized, {max_per_row} per row)')
        ax3.set_xlabel('Sequence Position')
        ax3.set_yticks([])
        ax3.set_xticks(range(0, max_per_row, max(1, max_per_row // 10)))



        message = analyze_top_regions(cam1, seq_letters, top_k=5)

        # Create the subplot at [3, 1]
        ax4 = fig.add_subplot(gs[3, col_idx])
        ax4.axis('off')  # Turn off axes

        # Display the text inside the axis
        ax4.text(
            0.5, 0.5,            # x, y in axis coordinates (0 = left/bottom, 1 = right/top)
            message,             # The text to display
            ha='center',         # horizontal alignment
            va='center',         # vertical alignment
            fontsize=10,         # adjust as needed
            family='monospace',  # makes alignment nicer for sequences
            wrap=True            # wrap long text
    )

    ax4 = fig.add_subplot(gs[4, 0])
    print(encoder_output_features.shape, np.array([encoder_output_features[0]]*3).shape)
    im = ax4.imshow([encoder_output_features[0]]*3, cmap='jet', alpha=0.8)
    ax4.set_title('Protein encoder extracted features')
    plt.colorbar(im, ax=ax4, shrink=0.5, aspect=10)
    
    ax4_2 = fig.add_subplot(gs[4, 1:3])
    ax4_2.imshow(gradaam_output)
    ax4_2.set_axis_off()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # make space for suptitle
    fig.suptitle(f"Model prediction: {prediction:.4f}. Ground truth: {y_label}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)

    print(f"Protein CAM visualization saved to: {save_path}")


def analyze_top_regions(grayscale_cam, protein_sequence_aa, top_k=5, amino_acid_vocab=None) -> str:
    """
    Identify and print the top-k most important regions
    """
    
    # Flatten CAM and sequence
    cam_1d = grayscale_cam.flatten()
    
    # Get top-k positions
    top_indices = np.argsort(cam_1d)[-top_k:][::-1]  # Descending order
    
    message = ''
    message += f"Top {top_k} most important positions:\n"
    
    for rank, idx in enumerate(top_indices, 1):
        if idx < len(protein_sequence_aa):
            aa = protein_sequence_aa[idx]
            weight = cam_1d[idx]
            if np.isclose(weight, 0.0):continue
            message += (f"{rank}. Position {idx:04d}: {aa} (attention: {weight:.4f})\n")

    return message


if __name__ == '__main__':
    main()

