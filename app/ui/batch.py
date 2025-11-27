import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def render_summary(metadata, results_df):
    st.header("📋 Batch Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples Analyzed", metadata.get("num_samples", "N/A"))
    with col2:
        st.metric("IG Steps", metadata.get("ig_steps", "N/A"))
    with col3:
        st.metric("Batch Size", metadata.get("batch_size", "N/A"))

    st.subheader("Aggregated Comparison Metrics")
    st.dataframe(results_df.describe().T)

def render_average_interactions(avg_grad_matrix, avg_ig_matrix):
    st.header("↔️ Average Atom-Residue Interactions")
    st.markdown("Compares the average interaction scores from Gradient and Integrated Gradients methods across the batch.")

    global_max = max(np.max(np.abs(avg_grad_matrix)), np.max(np.abs(avg_ig_matrix)))
    if global_max == 0: global_max = 1

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=go.Heatmap(z=avg_grad_matrix, colorscale='RdBu', zmin=-global_max, zmax=global_max))
        fig.update_layout(title="Average Gradient Interaction", xaxis_title="Atom Index", yaxis_title="Residue Index")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(data=go.Heatmap(z=avg_ig_matrix, colorscale='RdBu', zmin=-global_max, zmax=global_max))
        fig.update_layout(title="Average IG Interaction", xaxis_title="Atom Index", yaxis_title="Residue Index")
        st.plotly_chart(fig, use_container_width=True)

def render_embedding_maps(protein_tsne, ligand_tsne, protein_attrs, ligand_attrs):
    st.header("🗺️ 2D Embedding Maps (t-SNE)")
    st.markdown("Visualizes high-dimensional protein and ligand features in 2D, colored by their average attribution scores.")

    if protein_tsne is not None and protein_attrs is not None:
        fig = go.Figure(data=go.Scatter(
            x=protein_tsne[:, 0], y=protein_tsne[:, 1], mode='markers',
            marker=dict(size=5, color=protein_attrs, colorscale='Viridis', colorbar=dict(title="Attribution"), showscale=True),
            hovertext=[f"Attr: {a:.4f}" for a in protein_attrs]
        ))
        fig.update_layout(title="Protein Embedding t-SNE", xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Protein t-SNE data not available.")

    if ligand_tsne is not None and ligand_attrs is not None:
        fig = go.Figure(data=go.Scatter(
            x=ligand_tsne[:, 0], y=ligand_tsne[:, 1], mode='markers',
            marker=dict(size=5, color=ligand_attrs, colorscale='Plasma', colorbar=dict(title="Attribution"), showscale=True),
            hovertext=[f"Attr: {a:.4f}" for a in ligand_attrs]
        ))
        fig.update_layout(title="Ligand Embedding t-SNE", xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ligand t-SNE data not available.")

def render_feature_interaction_analysis(top_features, metadata):
    st.header("🔬 Embedding Interaction Analysis")
    k = metadata.get("k_top_interactions", "N/A")
    n = metadata.get("num_samples", "N/A")
    st.markdown(f"Frequency of embedding dimensions appearing in the top {k} interactions across {n} samples.")

    protein_feats = top_features['protein_features']
    ligand_feats = top_features['ligand_features']
    interaction_scores = top_features['interaction_scores']

    # Frequency Hotspots
    interaction_pairs = list(zip(protein_feats, ligand_feats))
    freq_matrix = np.zeros((96, 96))
    for p_idx, l_idx in interaction_pairs:
        if p_idx < 96 and l_idx < 96:
            freq_matrix[p_idx, l_idx] += 1
    
    fig_freq = go.Figure(data=go.Heatmap(z=freq_matrix, colorscale='Reds', colorbar=dict(title="Frequency")))
    fig_freq.update_layout(title="Feature Interaction Frequency Hotspots", xaxis_title="Ligand Feature", yaxis_title="Protein Feature")
    st.plotly_chart(fig_freq, use_container_width=True)

    # Average Score Landscape
    st.markdown("### Average Contribution to Prediction")
    st.markdown("This density map shows the average interaction score for each feature pair.")
    
    sum_matrix = np.zeros((96, 96))
    count_matrix = np.zeros((96, 96))
    for i in range(len(protein_feats)):
        p_idx, l_idx = protein_feats[i], ligand_feats[i]
        if p_idx < 96 and l_idx < 96:
            sum_matrix[p_idx, l_idx] += interaction_scores[i]
            count_matrix[p_idx, l_idx] += 1
            
    avg_score_matrix = np.divide(sum_matrix, count_matrix, where=count_matrix!=0)
    
    fig_avg = go.Figure(data=go.Contour(z=avg_score_matrix, colorscale='RdBu', zmid=0, colorbar=dict(title="Avg. Score")))
    fig_avg.update_layout(title="Average Interaction Score Landscape", xaxis_title="Ligand Feature", yaxis_title="Protein Feature")
    st.plotly_chart(fig_avg, use_container_width=True)
