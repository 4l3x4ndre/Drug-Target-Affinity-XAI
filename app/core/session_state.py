import streamlit as st

def init_session_state():
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    if 'gnn_explainer' not in st.session_state:
        st.session_state.gnn_explainer = None
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = ""
    if 'protein_input' not in st.session_state:
        st.session_state.protein_input = ""
    if 'sample_idx' not in st.session_state:
        st.session_state.sample_idx = 2
    if 'counter' not in st.session_state:
        st.session_state.counter = "0 / 1200"
