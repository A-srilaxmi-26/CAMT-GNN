from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EDGES_FILE = ROOT / "data" / "processed" / "food_web_edges.csv"
NODES_FILE = ROOT / "data" / "processed" / "food_web_nodes.csv"
NETWORKS_FILE = ROOT / "data" / "processed" / "food_web_networks.csv"

OUTPUT_FILE = ROOT / "data" / "processed" / "multitask_edges.csv"


edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)
networks = pd.read_csv(NETWORKS_FILE)


# --------------------------------------------------
# 1. Merge network-level information
# --------------------------------------------------

network_info = networks[
    [
        "network_id",
        "network_name",
        "network_date",
        "description",
        "latitude",
        "longitude"
    ]
].copy()

edges = edges.merge(
    network_info,
    on="network_id",
    how="left"
)


# --------------------------------------------------
# 2. Interaction-type target
# --------------------------------------------------

interaction_type_map = {
    "predation": 0,
    "herbivory": 1,
    "detritivore": 2
}

edges["interaction_type_target"] = (
    edges["interaction_type"]
    .map(interaction_type_map)
)


# --------------------------------------------------
# 3. Temporal target
# --------------------------------------------------

edges["network_year"] = pd.to_datetime(
    edges["network_date"],
    errors="coerce"
).dt.year


# --------------------------------------------------
# 4. Spatial target/context
# --------------------------------------------------

edges["has_spatial_data"] = (
    edges["latitude"].notna() |
    edges["longitude"].notna()
).astype(int)


# --------------------------------------------------
# 5. Save
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

edges.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n===== MULTI-TASK DATASET CREATED =====")
print(f"Rows: {len(edges)}")
print(f"Columns: {len(edges.columns)}")
print(f"Saved: {OUTPUT_FILE}")

print("\nInteraction-type targets:")
print(
    edges["interaction_type_target"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nNetwork years:")
print(
    edges.groupby("network_id")["network_year"]
    .first()
)

print("\nSpatial availability:")
print(
    edges.groupby("network_id")["has_spatial_data"]
    .first()
)

print("\n===== COMPLETE =====")