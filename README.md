# Explainable Drug-Target Affinity Prediction

This repository hosts the code and experiments for the project **"Explainable Drug-Target Affinity Prediction"**. 

This work addresses the "black box" nature of deep learning in drug discovery by applying a comprehensive framework of Explainable AI (XAI) techniques to the MGraphDTA model. The full project manuscript, detailing the methodology and extensive analysis, is available in the root directory of this repository.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/4l3x4ndre/drug-target-affinity-xai)

## 📄 Project Abstract

The prediction of drug-target affinity (DTA) is a cornerstone of modern drug discovery, yet the deep learning models that excel at this task are often "black boxes," hindering their adoption. This work addresses the critical need for interpretability by applying a comprehensive framework of Explainable AI (XAI) techniques to the MGraphDTA model. We leverage a suite of methods including **SHAP**, **GNNExplainer**, and **Integrated Gradients** to dissect the model's predictions.

Our analysis of the model's latent space reveals that its decision-making is highly context-dependent. We demonstrate the ability to generate plausible, fine-grained interaction hypotheses by identifying specific atom-amino acid pairs deemed important by the model. This XAI framework serves to build trust in DTA predictions and transforms black-box models into collaborative tools that can generate testable hypotheses.

## 🚀 Contributions

- **MGraphDTA Reproduction**: High-fidelity reproduction of the MGraphDTA model on the KIBA benchmark dataset.
- **Multi-Modal XAI**: Implementation of **LIME**, **KernelSHAP**, and **GNNExplainer** to analyze feature importance at atomic (ligand) and residue (protein) levels.
- **Interaction Analysis**: Gradient-based interaction matrices (using Integrated Gradients) to visualize specific atom-residue binding mechanisms.
- **Latent Space Analysis**: Exploration of the model's chemical manifold and context-dependent decision rules.
- **Interactive Dashboard**: A Streamlit-based web application for real-time exploration of model explanations.

<figure>
  <img src="docs/MGraphDTA_model.svg" alt="MGraphDTA model">
  <figcaption>
    Figure 1: MGraphDTA model architecture combining GNN for drug representation and CNN for protein sequences. The implementation used comes from the <a href="https://github.com/guaguabujianle/MGraphDTA">official repository</a>.
  </figcaption>
</figure>

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/4l3x4ndre/Drug-Target-Affinity-XAI
    cd drug-target-affinity-xai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset

This project primarily uses the **KIBA** dataset for regression tasks. 
All data used are publicly available. You can access them here:
- **Davis and KIBA**: [DeepDTA Repository](https://github.com/hkmztrk/DeepDTA/tree/master/data)
- **ToxCast**: [PADME Repository](https://github.com/simonfqy/PADME)

For training, ensure the data is placed in `MGraphDTA/regression/data/`.

## 🖥️ Usage & Training

The training process involves preprocessing the data into graph formats and then training the hybrid GNN-CNN model.

### 1. Preprocessing
Convert raw SMILES and protein sequences into graph and sequence representations.
```bash
cd MGraphDTA/regression
python preprocessing.py
```
*Note: This script processes data for both Davis and KIBA datasets.*

### 2. Training
Train the MGraphDTA model. The following command trains on the KIBA dataset and saves the model checkpoints.
```bash
python train.py --dataset kiba --save_model
```

**Parameters:**
- `--dataset`: Target dataset (`kiba` or `davis`).
- `--save_model`: Flag to save model checkpoints.
- `--lr`: Learning rate (default: `5e-4`).
- `--batch_size`: Batch size (default: `512`).

*Cluster Training:* For HPC environments, refer to the SLURM scripts provided in the `hpc/` directory (e.g., `hpc/train_regression.slurm`) for batch job submission examples.

### 3. Interactive Exploration (App)
Launch the Streamlit dashboard to visualize predictions and explanations:
```bash
streamlit run app/main.py
```
The app allows you to:
- Browse the validation dataset or input custom SMILES/Protein sequences.
- View **Attention Maps** for protein sequences.
- Visualize **GNN Explanations** (important atoms/bonds).
- Analyze **Drug-Target Interactions**.
- Run **SHAP** and **LIME** analyses.

### 4. Notebooks

Additional Jupyter notebooks are provided for other analyses in the `notebooks/` directory, including 

- latent space analysis (`notebooks/latent_space_exploration.ipynb` and `notebooks/latent_before-after_training.ipynb`)
- drug substructures analysis (`notebooks/batch_attention_analysis.ipynb`).

## 📚 References

### Original Model

This project reproduces and analyzes the **MGraphDTA** model. If you use the original architecture or dataset, please cite the original paper:

> Yang Z, Zhong W, Zhao L, Yu-Chian Chen C. **MGraphDTA: deep multiscale graph neural network for explainable drug-target binding affinity prediction.** *Chem Sci.* 2022 Jan 5;13(3):816-833. doi: 10.1039/d1sc05180f. PMID: 35173947; PMCID: PMC8768884.

The code base is adapted from the [official GitHub implementation](https://github.com/guaguabujianle/MGraphDTA) (MIT License).

This project also uses for [Advanced AI explainability for PyTorch GitHub repository](https://github.com/jacobgil/pytorch-grad-cam) for CNN explanations :

> Jacob Gildenblat, & contributors. (2021). **PyTorch library for CAM methods**, *GitHub*.

### This Work
For the XAI framework, implementation details, and specific findings presented here, please refer to the accompanying manuscript in this repository:

> **Amrani, A.** (2025). *Explainable Drug-Target Affinity Prediction*. Department of Computer Science, Norwegian University of Science and Technology.
> [WILL BE ADDED SOON]

## 📂 Repository Structure

- `app/`: Streamlit application code.
- `MGraphDTA/`: Core model implementation (adapted from official repo).
  - `regression/`: Training and preprocessing scripts.
- `xai/`: Implementation of Explainable AI methods (GNNExplainer, SHAP wrappers, etc.).
- `xai_dta/`: configuration and utility scripts for XAI experiments.
- `hpc/`: SLURM scripts for cluster training.
- `models/`: Saved model checkpoints.
- `data/`: Dataset storage.
- `GradCAM`: Grad-CAM implementation for CNN explanations.
