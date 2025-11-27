import sys
import os
import glob

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import streamlit as st
from core.model_loader import load_model, load_gnn_explainer
from core.session_state import init_session_state
from ui.home import render_home
from ui.tests import (
    render_attention_analysis,
    render_gnn_explanations,
    render_interaction_analysis,
    render_shap_explanations,
    render_lime_explanations,
    render_robustness_analysis,
    CACHE_DIR,
    ensure_main_explanation
)

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_all_cache():
    """Remove all cache files."""
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pt"))
    for f in cache_files:
        try:
            os.remove(f)
        except OSError as e:
            st.error(f"Error removing cache file {f}: {e}")
    print("All cache files cleared.")

def main():
    init_session_state()
    st.set_page_config(page_title="MGraphDTA Explainer", layout="wide", page_icon="🧬")

    # Initialize active test state if not present
    if 'active_test' not in st.session_state:
        st.session_state.active_test = None

    # Sidebar Navigation
    with st.sidebar:
        st.header("Navigation")
        page_selection = st.radio("Go to", ["Home", "Run Tests"])
        
        st.markdown("---")
        st.header("Settings")
        device = st.selectbox("Device", ["cpu", "cuda:0"], index=0)
        dataset_name='kiba'
        
        st.markdown("---")
        if st.button("🗑️ Clear All Cache"):
            clear_all_cache()
            st.success("Cache cleared!")
            st.rerun()

    # Load Model (Global)
    with st.spinner("Loading model..."):
        explainer = load_model(device, dataset_name=dataset_name)
        gnn_explainer_model = explainer.model
        gnn_explainer = load_gnn_explainer(gnn_explainer_model, device, dataset_name=dataset_name)
        st.session_state.explainer = explainer
        st.session_state.gnn_explainer = gnn_explainer

    # Render Pages
    if page_selection == "Home":
        render_home()
    
    elif page_selection == "Run Tests":
        st.title("🧪 Run Tests")
        
        # --- Input Section ---
        # Initialize widget state if not present
        if 'smiles_input_box' not in st.session_state:
            st.session_state.smiles_input_box = explainer.get_smile(st.session_state.sample_idx)
        if 'protein_input_box' not in st.session_state:
            st.session_state.protein_input_box = explainer.get_target(st.session_state.sample_idx)
        
        # Retrieve current inputs from widgets
        current_smiles = st.session_state.smiles_input_box
        current_protein = st.session_state.protein_input_box

        # Identify if this matches a dataset sample
        found_idx = explainer.get_dataset_idx(current_smiles, current_protein)
        
        # If valid match, sync the session state tracker
        if found_idx != -1:
            st.session_state.sample_idx = found_idx
            display_counter = f"{found_idx + 1} / {len(explainer)}"
            effective_sample_idx = found_idx
        else:
            display_counter = f"Custom / {len(explainer)}"
            effective_sample_idx = None

        # Navigation buttons
        nav_col1, nav_txt, nav_col2, _ = st.columns([.85,2,1,8])
        with nav_col1:
            if st.button("⬅️ Previous"):
                # Use last known sample_idx as baseline
                target_idx = max(0, st.session_state.sample_idx - 1)
                
                st.session_state.sample_idx = target_idx
                st.session_state.smiles_input_box = explainer.get_smile(target_idx)
                st.session_state.protein_input_box = explainer.get_target(target_idx)
                clear_all_cache() 
                st.rerun()

        with nav_col2:
            if st.button("Next ➡️"):
                # Use last known sample_idx as baseline
                target_idx = min(len(explainer) - 1, st.session_state.sample_idx + 1)
                
                st.session_state.sample_idx = target_idx
                st.session_state.smiles_input_box = explainer.get_smile(target_idx)
                st.session_state.protein_input_box = explainer.get_target(target_idx)
                clear_all_cache()
                st.rerun()

        with nav_txt:
            st.markdown(f"**Sample: {display_counter}**")

        col1, col2 = st.columns(2)
        with col1:
            smiles_input = st.text_input(
                "SMILES String",
                key="smiles_input_box",
                help="Enter a valid SMILES representation of the molecule"
            )

        with col2:
            protein_input = st.text_area(
                "Protein Sequence",
                key="protein_input_box",
                height=100,
                help="Enter the protein sequence (single letter amino acid codes)"
            )
        
        st.info("""
        Note: Sample used in the manuscript is number 3:
        - SMILES: `CC(C)(C(N)=O)n1cc(-c2cnc(N)c3c(-c4ccc(NC(=O)Nc5cccc(F)c5)cc4)csc23)cn1`
        - Protein: `MSGRPRTTSFAESCKPVQQPSAFGSMKVSRDKDGSKVTTVVATPGQGPDRPQEVSYTDTKVIGNGSFGVVYQAKLCDSGELVAIKKVLQDKRFKNRELQIMRKLDHCNIVRLRYFFYSSGEKKDEVYLNLVLDYVPETVYRVARHYSRAKQTLPVIYVKLYMYQLFRSLAYIHSFGICHRDIKPQNLLLDPDTAVLKLCDFGSAKQLVRGEPNVSYICSRYYRAPELIFGATDYTSSIDVWSAGCVLAELLLGQPIFPGDSGVDQLVEIIKVLGTPTREQIREMNPNYTEFKFPQIKAHPWTKVFRPRTPPEAIALCSRLLEYTPTARLTPLEACAHSFFDELRDPNVKLPNGRDTPALFNFTTQELSSNPPLATILIPPHARIQAAASTPTNATAASDANTGDRGQTNNAASASASNST`
        """, icon="ℹ️")
        st.markdown("---")
        
        # --- Prediction vs Ground Truth Section ---
        st.subheader("Model Prediction & Ground Truth")
        col_pred, col_true, col_diff = st.columns(3)
        # Get prediction
        try:
            data = None
            if smiles_input and protein_input:
                 # Use effective_sample_idx here to ensure we don't pull wrong labels for custom data
                 data = ensure_main_explanation(explainer, smiles_input, protein_input, effective_sample_idx)
            
            if data:
                output_dict = data['output']
                prediction_val = output_dict['prediction']
                
                with col_pred:
                    st.metric("Model Prediction", f"{prediction_val:.4f}")
                
                true_val = None
                try:
                    if 'true_affinity' in output_dict:
                        true_val = output_dict['true_affinity']
                except:
                    pass
                with col_true:
                    if true_val is not None:
                         st.metric("Ground Truth", f"{true_val:.4f}")
                    else:
                         st.metric("Ground Truth", "N/A")
                with col_diff:
                     if true_val is not None:
                         diff = prediction_val - true_val
                         st.metric("Difference (Pred - True)", f"{diff:.4f}")
                     else:
                         st.metric("Difference", "N/A")
        except Exception as e:
            st.warning(f"Could not display prediction summary: {e}")


        st.markdown("---")

        st.subheader("Select Test")
        st.warning('Please wait for all computations to finish before switching tests to avoid potential errors. (If a running man icon is visible at the top right, computations are still ongoing).',
                   icon="⚠️")
        
        # Test Selection Buttons
        t_col0, t_col1, t_col2, t_col3, t_col4, t_col5, t_col6 = st.columns(7)
        
        def set_test(test_name):
            st.session_state.active_test = test_name

        with t_col0:
            if st.button("Home", type="primary" if st.session_state.active_test is None else "secondary"):
                set_test(None)
        with t_col1:
            if st.button("Start Attention Analysis", type="primary" if st.session_state.active_test == "Attention" else "secondary"):
                set_test("Attention")
        with t_col2:
            if st.button("Start GNN Explanation", type="primary" if st.session_state.active_test == "GNN" else "secondary"):
                set_test("GNN")
        with t_col3:
            if st.button("Start Interaction Analysis", type="primary" if st.session_state.active_test == "Interaction" else "secondary"):
                set_test("Interaction")
        with t_col4:
            if st.button("Start SHAP Explanation", type="primary" if st.session_state.active_test == "SHAP" else "secondary"):
                set_test("SHAP")
        with t_col5:
            if st.button("Start LIME Explanation", type="primary" if st.session_state.active_test == "LIME" else "secondary"):
                set_test("LIME")
        with t_col6:
            if st.button("Start Robustness Analysis", type="primary" if st.session_state.active_test == "Robustness" else "secondary"):
                set_test("Robustness")


        st.markdown("---")

        # Render Active Test
        if st.session_state.active_test == "Attention":
            render_attention_analysis(explainer, smiles_input, protein_input, effective_sample_idx)
        elif st.session_state.active_test == "GNN":
            render_gnn_explanations(explainer, gnn_explainer, smiles_input, protein_input, effective_sample_idx)
        elif st.session_state.active_test == "Interaction":
            render_interaction_analysis(explainer, smiles_input, protein_input, effective_sample_idx)
        elif st.session_state.active_test == "SHAP":
            render_shap_explanations(explainer, smiles_input, protein_input, effective_sample_idx)
        elif st.session_state.active_test == "LIME":
            render_lime_explanations(explainer, smiles_input, protein_input, effective_sample_idx)
        elif st.session_state.active_test == "Robustness":
            render_robustness_analysis(explainer, smiles_input, protein_input, effective_sample_idx)
        elif st.session_state.active_test is None:
            st.info("Home. Select a test above to start.")

if __name__ == "__main__":
    set_seed(42)
    main()
