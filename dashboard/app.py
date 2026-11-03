from pathlib import Path
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RESULTS = ROOT / "data" / "results"
DATA_RAW = ROOT / "data" / "raw"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CAMT-GNN | Marine Ecosystem Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================

CSS_PATH = Path(__file__).resolve().parent / "assets" / "style.css"

if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ============================================================
# SAFE DATA LOADER
# ============================================================

def load_csv(filename, default=None):
    path = DATA_PROCESSED / filename

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass

    if default is None:
        return pd.DataFrame()

    return default


# ============================================================
# LOAD REAL PROJECT DATA
# ============================================================

nodes = load_csv("food_web_nodes.csv")
edges = load_csv("food_web_edges.csv")
networks = load_csv("food_web_networks.csv")

multitask = load_csv("multitask_edges.csv")
multitask_summary = load_csv("multitask_results_summary.csv")

climate_2472 = load_csv("network_2472_climate_spatial_features.csv")
climate_2474 = load_csv("network_2474_climate_spatial_features.csv")

# Results
baseline_results_path = DATA_RESULTS / "final_model_comparison.csv"

if baseline_results_path.exists():
    baseline_results = pd.read_csv(baseline_results_path)
else:
    baseline_results = pd.DataFrame()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <h1>🌊 CAMT-GNN</h1>
        <p>Marine Ecosystem Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Food Web",
        "Climate Monitor",
        "Temporal Analysis",
        "Species Intelligence",
        "Predictions",
        "Ecosystem Analytics",
        "Model Performance",
        "Explainability",
        "Data Sources",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption("Climate-Aware Multi-Modal Temporal Graph Neural Network")
st.sidebar.caption("Real-data research dashboard")


# ============================================================
# HELPER
# ============================================================

def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div>
                <h1>CAMT-GNN</h1>
                <p>
                    Climate-Aware Multi-Modal Temporal Graph Neural Network
                    for Marine Food Web Analysis and Ecosystem Monitoring
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Networks", len(networks), "Real Mangal food webs")

    with c2:
        metric_card("Nodes", len(nodes), "Marine taxa")

    with c3:
        metric_card("Interactions", len(edges), "Food-web relationships")

    with c4:
        metric_card(
            "Climate Networks",
            2 if not climate_2472.empty and not climate_2474.empty else 1,
            "NOAA OISST integrated",
        )

    st.markdown("---")

    st.markdown("## Research Pipeline")

    st.markdown(
        """
        <div class="pipeline">
            <div>Real Marine Data</div>
            <span>→</span>
            <div>Food Web Graph</div>
            <span>→</span>
            <div>Climate Features</div>
            <span>→</span>
            <div>GNN Models</div>
            <span>→</span>
            <div>Temporal Representation</div>
            <span>→</span>
            <div>Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Scientific note: the 2005 and 2007 Mangal records are separate "
        "network observations. They are not automatically treated as a "
        "continuous observation of the same food web."
    )


# ============================================================
# FOOD WEB
# ============================================================

elif page == "Food Web":

    st.title("🕸️ Marine Food Web")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Nodes", len(nodes))

    with c2:
        metric_card("Interactions", len(edges))

    with c3:
        metric_card("Networks", len(networks))

    st.markdown("---")

    if not networks.empty:
        st.subheader("Available Marine Networks")
        st.dataframe(networks, use_container_width=True)

    if not edges.empty and "type" in edges.columns:
        st.subheader("Interaction Types")

        counts = edges["type"].value_counts()

        st.bar_chart(counts)

        st.dataframe(
            counts.rename("count").reset_index(),
            use_container_width=True,
        )


# ============================================================
# CLIMATE MONITOR
# ============================================================

elif page == "Climate Monitor":

    st.title("🌡️ Climate Monitor")

    st.markdown(
        "Real NOAA OISST sea-surface-temperature features extracted "
        "for available Mangal network regions."
    )

    available = []

    if not climate_2472.empty:
        available.append(("2472", climate_2472))

    if not climate_2474.empty:
        available.append(("2474", climate_2474))

    if not available:
        st.warning("No climate feature files available.")
    else:

        for network_id, climate in available:

            st.markdown(f"## Network {network_id}")

            row = climate.iloc[0]

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "Mean SST",
                    f"{row['sst_mean']:.2f} °C",
                )

            with c2:
                metric_card(
                    "Minimum SST",
                    f"{row['sst_min']:.2f} °C",
                )

            with c3:
                metric_card(
                    "Maximum SST",
                    f"{row['sst_max']:.2f} °C",
                )

            with c4:
                metric_card(
                    "Grid Cells",
                    int(row["grid_cells"]),
                )

            st.dataframe(climate, use_container_width=True)


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

elif page == "Temporal Analysis":

    st.title("⏳ Temporal Analysis")

    st.markdown(
        """
        Temporal graph representation was constructed from the available
        dated Mangal network records.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        metric_card("2005 Network", "2472", "78 nodes / 1283 edges")

    with c2:
        metric_card("2007 Network", "2474", "78 nodes / 1328 edges")

    st.info(
        "The current temporal experiment produces temporal graph "
        "representations. It is not reported as a true temporal prediction "
        "experiment because the Mangal records are separate network records."
    )


# ============================================================
# SPECIES INTELLIGENCE
# ============================================================

elif page == "Species Intelligence":

    st.title("🐟 Species Intelligence")

    if nodes.empty:
        st.warning("Node dataset not available.")
    else:

        search = st.text_input(
            "Search species / taxa",
            placeholder="Enter species or taxon name...",
        )

        display = nodes.copy()

        if search and "species_name" in display.columns:
            mask = display["species_name"].astype(str).str.contains(
                search,
                case=False,
                na=False,
            )
            display = display[mask]

        st.metric("Matching taxa", len(display))

        st.dataframe(
            display,
            use_container_width=True,
            height=500,
        )


# ============================================================
# PREDICTIONS
# ============================================================

elif page == "Predictions":

    st.title("🔮 Predictions")

    st.info(
        "This page does not display fabricated prediction probabilities. "
        "Only verified trained-model outputs are shown."
    )

    if not multitask_summary.empty:
        st.subheader("Multi-Task Model Results")
        st.dataframe(
            multitask_summary,
            use_container_width=True,
        )
    else:
        st.warning("No verified prediction result table found.")


# ============================================================
# ECOSYSTEM ANALYTICS
# ============================================================

elif page == "Ecosystem Analytics":

    st.title("🌊 Ecosystem Analytics")

    if edges.empty:
        st.warning("Food-web data unavailable.")
    else:

        if "network_id" in edges.columns:

            network_counts = (
                edges.groupby("network_id")
                .size()
                .reset_index(name="interactions")
            )

            st.subheader("Interactions by Network")
            st.bar_chart(
                network_counts.set_index("network_id")
            )

        if "type" in edges.columns:

            st.subheader("Interaction Composition")

            type_counts = edges["type"].value_counts()

            st.bar_chart(type_counts)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    if baseline_results.empty:

        st.warning(
            "No final model comparison file found."
        )

    else:

        st.dataframe(
            baseline_results,
            use_container_width=True,
        )

        numeric_columns = [
            c for c in [
                "roc_auc",
                "average_precision",
                "precision",
                "recall",
                "f1",
            ]
            if c in baseline_results.columns
        ]

        if numeric_columns:

            st.subheader("Model Metrics")

            model_column = None

            for candidate in ["model", "Model"]:
                if candidate in baseline_results.columns:
                    model_column = candidate
                    break

            if model_column:
                chart_data = baseline_results.groupby(
                    model_column
                )[numeric_columns].mean()

                st.bar_chart(chart_data)


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.title("🧠 Explainability")

    shap_file = (
        DATA_RESULTS
        / "shap"
        / "feature_permutation_importance.csv"
    )

    if shap_file.exists():

        importance = pd.read_csv(shap_file)

        st.subheader("Feature Permutation Importance")

        st.caption(
            "This is permutation importance, not SHAP."
        )

        st.dataframe(
            importance,
            use_container_width=True,
        )

        if "importance" in importance.columns:

            top = importance.sort_values(
                "importance",
                ascending=False,
            ).head(15)

            st.bar_chart(
                top.set_index("feature")["importance"]
            )

    else:

        st.info(
            "No verified explainability artifact is available."
        )


# ============================================================
# DATA SOURCES
# ============================================================

elif page == "Data Sources":

    st.title("📚 Data Sources")

    st.markdown(
        """
        ### Mangal

        Marine food-web network and interaction data.

        ### NOAA OISST

        Sea-surface-temperature observations used for climate features.

        ### Project Outputs

        Processed datasets, graph files, model results and analysis
        artifacts are stored inside the project `data/` directory.
        """
    )

    st.markdown("## Project Data Status")

    files = [
        DATA_PROCESSED / "food_web_nodes.csv",
        DATA_PROCESSED / "food_web_edges.csv",
        DATA_PROCESSED / "food_web_networks.csv",
        DATA_PROCESSED / "multitask_edges.csv",
        DATA_RESULTS / "final_model_comparison.csv",
    ]

    status = []

    for file in files:
        status.append(
            {
                "File": str(file.relative_to(ROOT)),
                "Available": file.exists(),
            }
        )

    st.dataframe(
        pd.DataFrame(status),
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CAMT-GNN | Real-data marine ecosystem research dashboard"
)