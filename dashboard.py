
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ============================================================
# CAMT-GNN | Marine Ecosystem Intelligence
# ============================================================

ROOT = Path(__file__).resolve().parent

NETWORK_ID = 2472
NETWORK_NAME = "Northern California Current"
NETWORK_YEAR = 2005

CHECKPOINT = ROOT / "data/results/camt_gnn_2472_inference.pt"
GRAPH_PATH = ROOT / "data/processed/pyg/network_2472.pt"
NODES_CSV = ROOT / "data/processed/food_web_nodes.csv"
CLIMATE_CSV = ROOT / "data/processed/network_2472_climate_spatial_features.csv"

CLIMATE_COLUMNS = [
    "sst_mean",
    "sst_min",
    "sst_max",
    "sst_std",
    "sst_range",
    "sst_lat_gradient_mean",
    "sst_lon_gradient_mean",
]

TYPE_NAMES = {
    0: "Predation",
    1: "Herbivory",
    2: "Detritivore",
}

st.set_page_config(
    page_title="CAMT-GNN | Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111d;
        --panel: #0d1b2a;
        --panel2: #102235;
        --line: rgba(143, 190, 210, .15);
        --text: #e8f3f7;
        --muted: #8da6b8;
        --teal: #48e0c1;
        --blue: #60b9ff;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 75% -15%, rgba(27, 123, 151, .19), transparent 38%),
            linear-gradient(180deg, #07111d 0%, #081522 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 29, .75);
    }

    [data-testid="stSidebar"] {
        background: #091624;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    h1, h2, h3 {
        color: #effaff !important;
        letter-spacing: -.035em;
    }

    p, label, li {
        color: #c1d1dc;
    }

    .brand {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -.055em;
        color: #f2fbff;
    }

    .brand span {
        color: var(--teal);
    }

    .eyebrow {
        color: var(--teal);
        text-transform: uppercase;
        font-family: 'DM Mono', monospace;
        font-size: .70rem;
        letter-spacing: .15em;
        margin-bottom: 9px;
    }

    .hero {
        background:
            radial-gradient(ellipse at 88% 10%, rgba(72, 224, 193, .15), transparent 40%),
            linear-gradient(120deg, #10263a 0%, #0b1b2b 68%, #102c3c 100%);
        border: 1px solid rgba(103, 207, 207, .22);
        border-radius: 22px;
        padding: 32px 34px;
        margin: 6px 0 24px 0;
        position: relative;
        overflow: hidden;
    }

    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.08;
        margin: 8px 0 13px 0;
    }

    .hero p {
        max-width: 650px;
        line-height: 1.8;
        color: #b6ccd9;
        font-size: .95rem;
    }

    .hero-tag {
        display: inline-block;
        border: 1px solid rgba(72, 224, 193, .35);
        color: var(--teal);
        background: rgba(72, 224, 193, .07);
        border-radius: 50px;
        padding: 6px 11px;
        font-family: 'DM Mono', monospace;
        font-size: .68rem;
        letter-spacing: .05em;
    }

    .metric-card {
        background: linear-gradient(145deg, #102235, #0b1927);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 19px 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #92adbd;
        font-size: .76rem;
        margin-bottom: 14px;
    }

    .metric-value {
        color: #f1fbff;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -.05em;
    }

    .metric-foot {
        color: var(--teal);
        font-family: 'DM Mono', monospace;
        font-size: .68rem;
        margin-top: 8px;
    }

    .section-label {
        color: var(--teal);
        font-family: 'DM Mono', monospace;
        font-size: .69rem;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .panel {
        background: rgba(13, 27, 42, .85);
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 22px;
        margin: 10px 0;
    }

    .panel-title {
        color: #eaf7fc;
        font-weight: 750;
        font-size: 1.05rem;
        margin-bottom: 7px;
    }

    .muted {
        color: #8da6b8;
        font-size: .84rem;
        line-height: 1.7;
    }

    .status-dot {
        display: inline-block;
        height: 8px;
        width: 8px;
        background: #48e0c1;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px rgba(72, 224, 193, .55);
    }

    .result-number {
        font-size: 2.45rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.06em;
        color: var(--teal);
    }

    .small-label {
        color: #91aabb;
        font-size: .75rem;
        margin-bottom: 8px;
    }

    .stButton > button {
        background: linear-gradient(100deg, #48e0c1, #76ead4);
        color: #062019;
        border: 0;
        border-radius: 10px;
        font-weight: 800;
        min-height: 44px;
        transition: .2s ease;
    }

    .stButton > button:hover {
        background: #a0f5e5;
        color: #062019;
        border: 0;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: #0d1b2a;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #9bb3c2;
    }

    [data-testid="stMetricValue"] {
        color: #effaff;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #102235;
        border-color: #294355;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 9px 9px 0 0;
        color: #a7bdca;
        padding: 12px 18px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--teal) !important;
        border-bottom: 2px solid var(--teal);
    }

    hr {
        border-color: var(--line);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    .footer-note {
        border-top: 1px solid var(--line);
        padding-top: 18px;
        margin-top: 35px;
        color: #718c9e;
        font-family: 'DM Mono', monospace;
        font-size: .68rem;
        line-height: 1.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL ARCHITECTURE — MUST MATCH TRAINED CHECKPOINT
# ============================================================


class CAMTGNNInference(nn.Module):
    def __init__(
        self,
        input_dim,
        climate_dim=7,
        hidden_dim=64,
        climate_hidden=32,
        climate_out=16,
        num_types=3,
    ):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.climate_encoder = nn.Sequential(
            nn.Linear(climate_dim, climate_hidden),
            nn.ReLU(),
            nn.Linear(climate_hidden, climate_out),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + climate_out, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
        )

        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
        )

    def encode(self, x, edge_index, climate):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        climate_embedding = self.climate_encoder(climate)
        climate_embedding = climate_embedding.expand(h.size(0), -1)

        h = torch.cat([h, climate_embedding], dim=1)
        return self.fusion(h)

    def predict_pair(self, h, source_idx, target_idx):
        source = h[source_idx]
        target = h[target_idx]
        pair = torch.cat([source, target], dim=1)

        link_logit = self.link_head(pair).squeeze(-1)
        type_logits = self.type_head(pair)
        return link_logit, type_logits


# ============================================================
# VERIFIED DATA LOADING
# ============================================================


@st.cache_resource
def load_project():
    required = [CHECKPOINT, GRAPH_PATH, NODES_CSV, CLIMATE_CSV]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n" + "\n".join(missing)
        )

    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    graph = torch.load(
        GRAPH_PATH, map_location=device, weights_only=False
    )

    nodes = pd.read_csv(NODES_CSV)
    nodes = nodes[nodes["network_id"] == NETWORK_ID].reset_index(drop=True)

    climate_df = pd.read_csv(CLIMATE_CSV)
    climate_rows = climate_df[climate_df["network_id"] == NETWORK_ID]

    if nodes.empty or climate_rows.empty:
        raise ValueError("Network 2472 node or climate data is missing.")

    climate_row = climate_rows.iloc[0]
    climate_values = climate_row[CLIMATE_COLUMNS].astype(float).to_numpy()

    x = graph.x.float()
    edge_index = graph.edge_index.long()
    climate = torch.tensor(climate_values, dtype=torch.float32).unsqueeze(0)

    model = CAMTGNNInference(input_dim=x.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        embeddings = model.encode(x, edge_index, climate)

    # Convert graph edges to human-readable species pairs.
    # The project's graph uses node_from -> node_to orientation.
    if len(nodes) != x.shape[0]:
        raise ValueError(
            "Node metadata and graph feature counts do not match. "
            "Cannot safely map predictions to species."
        )

    edges = pd.DataFrame({
        "source_idx": edge_index[0].cpu().numpy(),
        "target_idx": edge_index[1].cpu().numpy(),
    })

    edges["Source / predator"] = edges["source_idx"].map(
        nodes["original_name"].to_dict()
    )
    edges["Target / prey"] = edges["target_idx"].map(
        nodes["original_name"].to_dict()
    )

    edges = edges.dropna(
        subset=["Source / predator", "Target / prey"]
    )

    return model, embeddings, edge_index, nodes, climate_values, edges


try:
    model, embeddings, edge_index, nodes, climate_values, edges = load_project()
except Exception as exc:
    st.error("CAMT-GNN could not load its project files.")
    st.exception(exc)
    st.stop()

species_names = nodes["original_name"].astype(str).tolist()


# ============================================================
# REUSABLE UI COMPONENTS
# ============================================================


def metric_card(label, value, foot):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(kicker, title, description=None):
    st.markdown(
        f'<div class="section-label">{kicker}</div>',
        unsafe_allow_html=True,
    )
    st.subheader(title)
    if description:
        st.markdown(
            f'<div class="muted">{description}</div>',
            unsafe_allow_html=True,
        )


def show_climate_cards():
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("MEAN SEA TEMPERATURE", f"{climate_values[0]:.2f}°C", "NOAA OISST")

    with c2:
        metric_card("MINIMUM SST", f"{climate_values[1]:.2f}°C", "Observed grid minimum")

    with c3:
        metric_card("MAXIMUM SST", f"{climate_values[2]:.2f}°C", "Observed grid maximum")

    with c4:
        metric_card("TEMPERATURE RANGE", f"{climate_values[4]:.2f}°C", "Maximum − minimum")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">CAMT<span>·</span>GNN</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">MARINE INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Food Web Explorer",
            "Interaction Prediction",
            "Climate Intelligence",
            "Species Explorer",
            "Model & Data",
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="panel">
            <div class="section-label">ACTIVE STUDY</div>
            <div class="panel-title">Northern California Current</div>
            <div class="muted">Mangal network 2472 · 2005</div>
            <br>
            <span class="status-dot"></span>
            <span style="color:#48e0c1;font-size:.8rem">Model loaded</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">DATA SOURCES<br>Mangal · NOAA OISST</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# COMMON HERO
# ============================================================

page_copy = {
    "Overview": (
        "ECOSYSTEM INTELLIGENCE / 01",
        "Read the ocean as a living network.",
        "Explore a real marine food web, examine its observed sea-surface temperatures, "
        "and run pairwise interaction predictions with the trained CAMT-GNN model.",
    ),
    "Food Web Explorer": (
        "NETWORK ANALYTICS / 02",
        "Explore who connects to whom.",
        "Inspect recorded directed interactions in the selected Mangal network.",
    ),
    "Interaction Prediction": (
        "MODEL WORKSPACE / 03",
        "Test a species interaction.",
        "Choose two real nodes and inspect the trained model's link score and interaction-type output.",
    ),
    "Climate Intelligence": (
        "CLIMATE MONITOR / 04",
        "Read the temperature signal.",
        "Review the extracted NOAA OISST spatial summary for the selected marine region and date.",
    ),
    "Species Explorer": (
        "BIODIVERSITY / 05",
        "Meet the nodes in this food web.",
        "Search species and functional groups and inspect their recorded network connections.",
    ),
    "Model & Data": (
        "RESEARCH TRANSPARENCY / 06",
        "Inspect the evidence behind the interface.",
        "Review model provenance, dataset scope, and limitations before interpreting predictions.",
    ),
}

kicker, headline, description = page_copy[page]

st.markdown(
    f"""
    <div class="hero">
        <span class="hero-tag">● LIVE LOCAL MODEL · NETWORK {NETWORK_ID}</span>
        <div class="eyebrow" style="margin-top:22px">{kicker}</div>
        <h1>{headline}</h1>
        <p>{description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("FOOD-WEB NODES", f"{len(nodes):,}", "Mangal node records")

    with c2:
        metric_card("RECORDED EDGES", f"{edge_index.shape[1]:,}", "Directed interactions")

    with c3:
        metric_card("MEAN SST", f"{climate_values[0]:.2f}°C", "NOAA OISST · 2005-06-01")

    with c4:
        metric_card("SST RANGE", f"{climate_values[4]:.2f}°C", "Extracted spatial subset")

    st.write("")
    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        section_heading(
            "NETWORK SNAPSHOT",
            "Recorded interaction types",
            "Counts are computed from the actual edge file for this network.",
        )

        if "type" in pd.read_csv(ROOT / "data/processed/food_web_edges.csv", nrows=0).columns:
            all_edges = pd.read_csv(ROOT / "data/processed/food_web_edges.csv")
            if "network_id" in all_edges.columns:
                all_edges = all_edges[all_edges["network_id"] == NETWORK_ID]

            if "type" in all_edges.columns:
                counts = all_edges["type"].fillna("Unknown").value_counts()
                st.bar_chart(counts)
                st.dataframe(
                    counts.rename_axis("Interaction type")
                    .reset_index(name="Recorded edges"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Interaction-type field was not available.")
        else:
            st.info("Could not find the interaction-type field.")

    with right:
        section_heading(
            "CLIMATE CONTEXT",
            "Sea-surface temperature",
            "Spatial statistics extracted from the regional NOAA OISST subset.",
        )
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            f'<div class="small-label">REGIONAL MEAN</div>'
            f'<div class="result-number">{climate_values[0]:.2f}<span style="font-size:1rem"> °C</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(
            min(max((climate_values[0] / 30), 0.0), 1.0),
            text="Mean SST shown on a 0–30°C display scale",
        )
        st.markdown(
            f'<div class="muted">Minimum {climate_values[1]:.2f}°C · '
            f'Maximum {climate_values[2]:.2f}°C</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    section_heading(
        "QUICK ACCESS",
        "Choose your next investigation",
        "Use the sidebar to open a dedicated workspace.",
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown(
            '<div class="panel"><div class="panel-title">01 / Food web</div>'
            '<div class="muted">Search recorded predator–prey connections and inspect edge records.</div></div>',
            unsafe_allow_html=True,
        )

    with q2:
        st.markdown(
            '<div class="panel"><div class="panel-title">02 / Prediction</div>'
            '<div class="muted">Run the trained model on a selected source–target pair.</div></div>',
            unsafe_allow_html=True,
        )

    with q3:
        st.markdown(
            '<div class="panel"><div class="panel-title">03 / Climate</div>'
            '<div class="muted">Inspect real regional SST summary features and their provenance.</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 2 — FOOD WEB EXPLORER
# ============================================================

elif page == "Food Web Explorer":
    section_heading(
        "DIRECTED NETWORK",
        "Food-web connections",
        "The table lists recorded graph edges. The source/target orientation follows the project's Mangal graph construction.",
    )

    st.caption(
        f"Network {NETWORK_ID} · {len(nodes)} nodes · {len(edges)} directed edges"
    )

    search_col, sort_col = st.columns([2, 1])

    with search_col:
        species_filter = st.text_input(
            "Filter by species or functional group",
            placeholder="e.g. copepods, coho, phytoplankton",
        )

    with sort_col:
        view_size = st.selectbox(
            "Rows to display",
            [25, 50, 100, 250],
            index=1,
        )

    view = edges.copy()

    if species_filter.strip():
        term = species_filter.strip().lower()
        mask = (
            view["Source / predator"].str.lower().str.contains(term, na=False)
            | view["Target / prey"].str.lower().str.contains(term, na=False)
        )
        view = view[mask]

    st.dataframe(
        view[["Source / predator", "Target / prey"]].head(view_size),
        use_container_width=True,
        hide_index=True,
        height=450,
    )

    st.download_button(
        "Download displayed edge records",
        data=view[["Source / predator", "Target / prey"]].to_csv(index=False),
        file_name=f"camt_gnn_network_{NETWORK_ID}_edges.csv",
        mime="text/csv",
    )

    st.markdown(
        '<div class="muted">Only recorded graph edges are shown here. '
        'An unrecorded pair is not automatically proof that the ecological interaction never occurs.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 3 — INTERACTION PREDICTION
# ============================================================

elif page == "Interaction Prediction":
    section_heading(
        "PAIRWISE INFERENCE",
        "Configure a prediction",
        "Select nodes from the real Network 2472 node table.",
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Species pair</div>', unsafe_allow_html=True)

        predator = st.selectbox(
            "Source / predator",
            species_names,
            index=species_names.index("PiscF1a: coho")
            if "PiscF1a: coho" in species_names else 0,
            key="predict_source",
        )

        prey = st.selectbox(
            "Target / prey",
            species_names,
            index=0,
            key="predict_target",
        )

        st.markdown(
            '<div class="muted">The trained model uses the saved graph, '
            'the selected node embeddings, and the recorded network climate context.</div>',
            unsafe_allow_html=True,
        )
        run_prediction = st.button(
            "Run CAMT-GNN inference",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Climate context used</div>', unsafe_allow_html=True)

        for label, value in zip(CLIMATE_COLUMNS, climate_values):
            st.write(f"**{label.replace('_', ' ').title()}** · {value:.4f}")

        st.markdown(
            '<div class="muted">These values come from the stored Network 2472 NOAA OISST feature file. '
            'They are not editable scenario inputs.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if run_prediction:
        source_idx = species_names.index(predator)
        target_idx = species_names.index(prey)

        with torch.no_grad():
            link_logit, type_logits = model.predict_pair(
                embeddings,
                torch.tensor([source_idx], dtype=torch.long),
                torch.tensor([target_idx], dtype=torch.long),
            )

            link_score = torch.sigmoid(link_logit).item()
            type_probs = torch.softmax(type_logits, dim=1)[0].cpu().numpy()

        type_idx = int(np.argmax(type_probs))
        predicted_type = TYPE_NAMES[type_idx]

        recorded_pair = bool(
            (
                (edge_index[0] == source_idx)
                & (edge_index[1] == target_idx)
            ).any()
        )

        st.markdown("---")
        section_heading(
            "MODEL OUTPUT",
            f"{predator}  →  {prey}",
            "These are outputs from the saved model, not manually assigned values.",
        )

        a, b, c = st.columns(3)

        with a:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">SIGMOID LINK SCORE</div>'
                f'<div class="metric-value">{link_score * 100:.2f}%</div>'
                f'<div class="metric-foot">Model output · not calibrated</div></div>',
                unsafe_allow_html=True,
            )

        with b:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">TOP INTERACTION CLASS</div>'
                f'<div class="metric-value" style="font-size:1.35rem">{predicted_type}</div>'
                f'<div class="metric-foot">Highest softmax class</div></div>',
                unsafe_allow_html=True,
            )

        with c:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">RECORDED EDGE</div>'
                f'<div class="metric-value" style="font-size:1.35rem">'
                f'{"YES" if recorded_pair else "NO"}</div>'
                f'<div class="metric-foot">Observed dataset lookup</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("#### Interaction-type score distribution")

        type_df = pd.DataFrame({
            "Interaction type": [TYPE_NAMES[i] for i in range(3)],
            "Softmax score (%)": (type_probs * 100).round(3),
        }).set_index("Interaction type")

        st.bar_chart(type_df)

        st.markdown(
            '<div class="panel"><div class="panel-title">How to interpret this result</div>'
            '<div class="muted">The sigmoid link score is the model output for this pair; it has not been '
            'established as a calibrated ecological probability. The top interaction class is the class '
            'with the largest softmax score. A high class score is not the same as validated accuracy. '
            'An absent recorded edge may reflect missing or unrecorded data.</div></div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "Download this prediction",
            data=pd.DataFrame([{
                "network_id": NETWORK_ID,
                "source": predator,
                "target": prey,
                "sigmoid_link_score": link_score,
                "predicted_type": predicted_type,
                "type_softmax_score": float(type_probs[type_idx]),
                "edge_recorded_in_dataset": recorded_pair,
                "sst_mean": float(climate_values[0]),
            }]).to_csv(index=False),
            file_name="camt_gnn_prediction.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE 4 — CLIMATE INTELLIGENCE
# ============================================================

elif page == "Climate Intelligence":
    section_heading(
        "NOAA OISST",
        "Regional sea-surface temperature",
        "Spatial summary calculated from the Network 2472 subset for 1 June 2005.",
    )

    show_climate_cards()

    st.write("")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Climate feature profile</div>', unsafe_allow_html=True)

        climate_profile = pd.DataFrame({
            "Feature": CLIMATE_COLUMNS,
            "Value": climate_values,
        }).set_index("Feature")

        st.bar_chart(climate_profile)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Measurement summary</div>', unsafe_allow_html=True)

        climate_table = pd.DataFrame({
            "Feature": CLIMATE_COLUMNS,
            "Value": climate_values,
        })

        st.dataframe(
            climate_table,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            '<div class="muted">The SST range is maximum minus minimum across the extracted grid. '
            'Gradient features are the spatial-gradient summaries produced by the existing preprocessing script.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "This view represents one dated spatial climate snapshot, not a continuous real-time ocean sensor feed."
    )


# ============================================================
# PAGE 5 — SPECIES EXPLORER
# ============================================================

elif page == "Species Explorer":
    section_heading(
        "NODE CATALOGUE",
        "Species and functional groups",
        "Search real node names from Network 2472.",
    )

    search = st.text_input(
        "Search nodes",
        placeholder="Enter a species or functional group name...",
    )

    filtered_nodes = nodes.copy()

    if search.strip():
        term = search.strip().lower()
        searchable = (
            filtered_nodes["original_name"].astype(str).str.lower().str.contains(term, na=False)
        )

        if "species_name" in filtered_nodes.columns:
            searchable |= (
                filtered_nodes["species_name"].astype(str).str.lower().str.contains(term, na=False)
            )

        filtered_nodes = filtered_nodes[searchable]

    st.caption(f"{len(filtered_nodes)} matching nodes")

    display_columns = [
        c for c in [
            "original_name",
            "species_name",
            "node_level",
            "taxonomy_name",
        ]
        if c in filtered_nodes.columns
    ]

    st.dataframe(
        filtered_nodes[display_columns],
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    if not filtered_nodes.empty:
        selected_name = st.selectbox(
            "Inspect a node",
            filtered_nodes["original_name"].astype(str).tolist(),
        )

        idx = species_names.index(selected_name)

        outgoing = edges[edges["source_idx"] == idx]
        incoming = edges[edges["target_idx"] == idx]

        x1, x2, x3 = st.columns(3)

        x1.metric("Outgoing connections", len(outgoing))
        x2.metric("Incoming connections", len(incoming))
        x3.metric("Total recorded connections", len(outgoing) + len(incoming))

        left, right = st.columns(2)

        with left:
            st.markdown("#### Outgoing edges")
            st.dataframe(
                outgoing[["Source / predator", "Target / prey"]].head(100),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.markdown("#### Incoming edges")
            st.dataframe(
                incoming[["Source / predator", "Target / prey"]].head(100),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# PAGE 6 — MODEL & DATA TRANSPARENCY
# ============================================================

elif page == "Model & Data":
    section_heading(
        "REPRODUCIBILITY",
        "Model and data provenance",
        "The interface uses the files currently saved in your CAMT-GNN project.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Model architecture</div>
            <div class="muted">
                Two graph-convolution layers → climate encoder → feature fusion →
                link-prediction head and interaction-type head.
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">Food-web source</div>
            <div class="muted">
                Mangal network 2472 · Northern California Current · 2005.
                The graph loaded by this dashboard contains the saved network nodes and directed edges.
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">Climate source</div>
            <div class="muted">
                NOAA OISST subset for 1 June 2005, extracted for the Network 2472 region.
                The dashboard uses the saved spatial climate feature CSV.
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">Scope of inference</div>
            <div class="muted">
                This checkpoint was trained for Network 2472. Predictions are pairwise model outputs
                under this graph and climate context; they are not a validated universal marine predictor.
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">Research limitations</div>
            <div class="muted">
                The sigmoid link score has not been demonstrated to be calibrated. Softmax values are
                relative class scores, not proven ecological certainty. The climate input is a single
                dated regional snapshot. An unrecorded edge is not proof that an interaction is absent.
                Climate benefit must not be claimed without controlled evaluation supporting it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Loaded project files")

    file_status = pd.DataFrame({
        "File": [
            str(CHECKPOINT.relative_to(ROOT)),
            str(GRAPH_PATH.relative_to(ROOT)),
            str(NODES_CSV.relative_to(ROOT)),
            str(CLIMATE_CSV.relative_to(ROOT)),
        ],
        "Available": [
            p.exists()
            for p in [CHECKPOINT, GRAPH_PATH, NODES_CSV, CLIMATE_CSV]
        ],
    })

    st.dataframe(
        file_status,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        CAMT-GNN · CLIMATE-AWARE MARINE FOOD WEB ANALYSIS<br>
        RESEARCH PROTOTYPE · MANGAL + NOAA OISST · NETWORK 2472<br>
        Predictions require scientific validation before ecological or management use.
    </div>
    """,
    unsafe_allow_html=True,
)
