import streamlit as st
from xai.visualization.gnn_visual import create_gnn_summary_cards

def display_gnn_summary_streamlit(gnn_results):
    """
    Display GNN explanation summary in Streamlit format.
    """
    summary = create_gnn_summary_cards(gnn_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Method", value="GNN Explainer")
    with col2:
        st.metric(label="Prediction", value=f"{summary['prediction']:.4f}")
    with col3:
        st.metric(label="Atoms", value=summary['num_atoms'])
    with col4:
        st.metric(label="Bonds", value=summary['num_bonds'])
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(label="Mean Node Imp.", value=f"{summary['mean_node_importance']:.4f}")
    with col6:
        st.metric(label="Max Node Imp.", value=f"{summary['max_node_importance']:.4f}", help=f"Atom {summary['top_node_index']} (most important)")
    with col7:
        st.metric(label="Mean Edge Imp.", value=f"{summary['mean_edge_importance']:.4f}")
    with col8:
        st.metric(label="Max Edge Imp.", value=f"{summary['max_edge_importance']:.4f}", help=f"Bond {summary['top_edge_source']}-{summary['top_edge_target']} (most important)")


import streamlit as st
from xai.visualization.gnn_visual import create_gnn_summary_cards
import os
import matplotlib

def display_gnn_summary_streamlit(gnn_results):
    """
    Display GNN explanation summary in Streamlit format.
    """
    summary = create_gnn_summary_cards(gnn_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Method", value="GNN Explainer")
    with col2:
        st.metric(label="Prediction", value=f"{summary['prediction']:.4f}")
    with col3:
        st.metric(label="Atoms", value=summary['num_atoms'])
    with col4:
        st.metric(label="Bonds", value=summary['num_bonds'])
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(label="Mean Node Imp.", value=f"{summary['mean_node_importance']:.4f}")
    with col6:
        st.metric(label="Max Node Imp.", value=f"{summary['max_node_importance']:.4f}", help=f"Atom {summary['top_node_index']} (most important)")
    with col7:
        st.metric(label="Mean Edge Imp.", value=f"{summary['mean_edge_importance']:.4f}")
    with col8:
        st.metric(label="Max Edge Imp.", value=f"{summary['max_edge_importance']:.4f}", help=f"Bond {summary['top_edge_source']}-{summary['top_edge_target']} (most important)")


def gnn_main_visualisation_st(gnn_figures, gnn_output, ligand_img):
    st.header("Molecular Graph Explanation")
    st.markdown("**GNN Explainer** identifies the most important atoms and bonds in the molecular graph for the prediction.")
    
    display_gnn_summary_streamlit(gnn_output)
    
    if gnn_output['mol_visualization']:
        mol_viz = gnn_output['mol_visualization']
        st.markdown("**Molecular Structure with Importance Highlighting:**")
        st.markdown(f"*Red colors indicate high importance, blue colors indicate low importance*")
        svg_data = mol_viz['svg_data']

        col1, col2 = st.columns(2)
        with col1:
            st.image(ligand_img, caption="Molecule with Atom Importance as given by Grad-AAM from reference study (Yang et al. 2022).", width=400)
        with col2:
            st.image(svg_data, caption="GNNExplainer-based Importance", width=400)
            if st.button("Save GNN Molecule as SVG"):
                output_dir = "results/gnn_analysis"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filepath = os.path.join(output_dir, "gnn_molecule.svg")
                with open(filepath, "w") as f:
                    f.write(svg_data)
                st.success(f"Saved to {filepath}")


    col1, col2 = st.columns(2)
    with col1:
        if gnn_output['top_nodes']:
            st.markdown("**Top Important Atoms from GNNExplainer:**")
            for i, node in enumerate(gnn_output['top_nodes'][:5]):
                st.markdown(f"{i+1}. Atom {node['label']}: {node['importance']:.4f}")
    with col2:
        if gnn_output['top_edges']:
            st.markdown("**Top Important Bonds from GNNExplainer:**")
            for i, edge in enumerate(gnn_output['top_edges'][:5]):
                st.markdown(f"{i+1}. Bond {edge['source']} - {edge['target']}: {edge['importance']:.4f}")
    
    if gnn_figures:
        st.header("GNN Explanation Visualizations")
        
        matplotlib_plots = ['node_importance', 'edge_importance', 'node_distribution', 'edge_distribution']


        for name, fig in gnn_figures.items():
            if fig is None:
                continue

            if name in matplotlib_plots:
                st.pyplot(fig)
                if st.button(f"Save {name.replace('_', ' ').title()} as SVG", key=f"save_gnn_{name}"):
                    output_dir = "results/gnn_analysis"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    filepath = os.path.join(output_dir, f"gnn_{name}.svg")
                    try:
                        fig.savefig(filepath, format='svg', bbox_inches='tight')
                        st.success(f"Saved to {filepath}")
                    except Exception as e:
                        st.error(f"Failed to save SVG. Error: {e}")
            else:
                st.plotly_chart(fig, use_container_width=True)


def render_gnn_explanation_streamlit(fig_feat, fig_graph):
    st.subheader("🔍 GNN Explanation Results")

    if fig_feat:
        st.pyplot(fig_feat)
        if st.button("Save Feature Importance as SVG"):
            output_dir = "results/gnn_analysis"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, "gnn_feature_importance.svg")
            try:
                fig_feat.savefig(filepath, format='svg', bbox_inches='tight')
                st.success(f"Saved to {filepath}")
            except Exception as e:
                st.error(f"Failed to save SVG. Error: {e}")
    else:
        st.info("No feature importance data available from explainer.")

    # st.markdown("---")
    # st.subheader("Computational Graph Explanation")
    #
    # if fig_graph:
    #     st.plotly_chart(fig_graph, use_container_width=True)
    #     st.caption("Node size and color, and edge width and opacity, represent importance scores.")
    # else:
    #     st.warning("No graph to display with the current settings.")
