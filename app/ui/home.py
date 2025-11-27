import streamlit as st

def render_home():
    st.title("📖 How to Guide")
    
    st.markdown("""
    Welcome to the **MGraphDTA Explainer**! This tool is designed to help you understand the predictions of the MGraphDTA model for Drug-Target Affinity.

    See [GitHub repository](https://github.com/4l3x4ndre/Drug-Target-Affinity-XAI).

    ## Getting Started
    
    1.  **Navigate to 'Run Tests'**: Use the sidebar to switch to the testing environment.
    2.  **Select Input Data**:
        *   You can use the **Previous/Next** buttons to cycle through the validation dataset.
        *   Alternatively, you can manually paste a **SMILES string** (for the drug) and a **Protein Sequence** (for the target).
    3.  **Choose an Analysis**:
        *   Once your inputs are ready, click on one of the "Start ... Test" buttons.
        *   Only one test can be active at a time.
    
    ## Available Tests
    
    *   **🔬 Attention Analysis**: Visualizes the attention weights of the model on the protein sequence and the ligand atoms. Also includes the fraction of importance for protein residues. Helps understand which parts of the molecule and protein the model focuses on.
    *   **🧠 GNN Explanations**: Uses Graph Neural Network explainers to highlight important edges and features in the molecular graph.
    *   **🤝 Interaction Analysis**: Analyzes the specific interactions between the drug and the target.
    *   **📊 SHAP Explanations**: Uses SHAP (SHapley Additive exPlanations) values to quantify the contribution of each feature to the prediction.
    *   **🍋 LIME Explanations**: Uses LIME (Local Interpretable Model-agnostic Explanations) to approximate the model locally and explain predictions.
    *   **🛡️ Robustness Analysis**: Evaluates the model's sensitivity to masking important amino acids in the protein sequence (masking robustness).
    
    ## Tips
    
    *   **Caching**: The app caches results to speed up subsequent views. If you change inputs, the relevant cache is automatically invalidated.
    *   **Device**: You can switch between CPU and GPU in the sidebar settings (if available).

    ## Materials

    MGraphDTA model is coming from the paper : 

    > Yang Z, Zhong W, Zhao L, Yu-Chian Chen C. MGraphDTA: deep multiscale graph neural network for explainable drug-target binding affinity prediction. Chem Sci. 2022 Jan 5;13(3):816-833. doi: 10.1039/d1sc05180f. PMID: 35173947; PMCID: PMC8768884.
     
    This project uses an adapted implementation of the [official GitHub implementation](https://github.com/guaguabujianle/MGraphDTA) distributed under the MIT License. 
    Dataset preparation and model loading code is adapted from the original repository.

    This project is not affiliated with the original authors.

    This project also uses [Advanced AI explainability for PyTorch GitHub repository](https://github.com/jacobgil/pytorch-grad-cam) for CNN explanations:

    > Jacob Gildenblat, & contributors. (2021). PyTorch library for CAM methods, *GitHub*.
    """)
