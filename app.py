from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CAMT-GNN Marine Ecosystem Monitor",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 CAMT-GNN")
st.subheader(
    "Climate-Aware Multi-Modal Temporal Graph Neural Network "
    "for Marine Food Web Analysis"
)

# -----------------------------
# Load datasets
# -----------------------------

edges_path = ROOT / "data/processed/food_web_edges.csv"
nodes_path = ROOT / "data/processed/food_web_nodes.csv"
networks_path = ROOT / "data/processed/food_web_networks.csv"
results_path = ROOT / "data/results/final_model_comparison.csv"

edges = pd.read_csv(edges_path)
nodes = pd.read_csv(nodes_path)
networks = pd.read_csv(networks_path)
results = pd.read_csv(results_path)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Network Selection")

network_ids = sorted(edges["network_id"].unique())

selected_network = st.sidebar.selectbox(
    "Select Marine Network",
    network_ids
)

network_edges = edges[
    edges["network_id"] == selected_network
]

network_nodes = nodes[
    nodes["network_id"] == selected_network
]

# -----------------------------
# Overview metrics
# -----------------------------

st.header("Marine Food Web Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Network",
    selected_network
)

c2.metric(
    "Nodes",
    len(network_nodes)
)

c3.metric(
    "Interactions",
    len(network_edges)
)

c4.metric(
    "Interaction Types",
    network_edges["interaction_type"].nunique()
)

# -----------------------------
# Interaction distribution
# -----------------------------

st.header("Food Web Interactions")

interaction_counts = (
    network_edges["interaction_type"]
    .value_counts()
    .rename_axis("Interaction Type")
    .reset_index(name="Count")
)

st.bar_chart(
    interaction_counts.set_index("Interaction Type")
)

# -----------------------------
# Network data
# -----------------------------

st.header("Network Information")

network_info = networks[
    networks["network_id"] == selected_network
]

st.dataframe(
    network_info,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Node data
# -----------------------------

st.header("Marine Species / Node Data")

st.dataframe(
    network_nodes,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Model comparison
# -----------------------------

st.header("CAMT-GNN Model Evaluation")

model_summary = (
    results.groupby("model")[
        [
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1"
        ]
    ]
    .mean()
    .sort_values("roc_auc", ascending=False)
)

st.dataframe(
    model_summary.round(4),
    use_container_width=True
)

st.bar_chart(
    model_summary["roc_auc"]
)

# -----------------------------
# Climate information
# -----------------------------

st.header("Climate Information")

climate_file = (
    ROOT /
    f"data/processed/network_{selected_network}_climate_spatial_features.csv"
)

if climate_file.exists():

    climate = pd.read_csv(climate_file)

    st.dataframe(
        climate,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Climate data is not currently available for this network."
    )

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "CAMT-GNN | Real Mangal marine food-web data | "
    "Climate-aware graph learning research prototype"
)