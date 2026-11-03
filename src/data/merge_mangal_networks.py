from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANGAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mangal"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


NETWORK_IDS = [
    2472,
    2474,
    5028,
    5027,
    5029,
]


# ============================================================
# HELPER
# ============================================================

def get_id(value):
    """Extract an ID whether Mangal stores a value as an ID or object."""

    if isinstance(value, dict):
        return value.get("id")

    return value


# ============================================================
# LOAD NETWORKS
# ============================================================

all_interactions = []
all_nodes = []
network_metadata = []


for network_id in NETWORK_IDS:

    network_dir = MANGAL_DIR / str(network_id)

    print("=" * 70)
    print(f"PROCESSING NETWORK {network_id}")
    print("=" * 70)

    with open(
        network_dir / "network.json",
        encoding="utf-8"
    ) as f:
        network = json.load(f)

    with open(
        network_dir / "nodes.json",
        encoding="utf-8"
    ) as f:
        nodes = json.load(f)

    with open(
        network_dir / "interactions.json",
        encoding="utf-8"
    ) as f:
        interactions = json.load(f)

    # --------------------------------------------------------
    # Network metadata
    # --------------------------------------------------------

    geom = network.get("geom") or {}

    coordinates = geom.get("coordinates", [None, None])

    network_metadata.append(
        {
            "network_id": network.get("id"),
            "network_name": network.get("name"),
            "network_date": network.get("date"),
            "description": network.get("description"),
            "latitude": (
                coordinates[1]
                if len(coordinates) > 1
                else None
            ),
            "longitude": (
                coordinates[0]
                if len(coordinates) > 0
                else None
            ),
        }
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    for node in nodes:

        taxonomy = node.get("taxonomy") or {}

        all_nodes.append(
            {
                "network_id": network_id,
                "node_id": node.get("id"),
                "original_name": node.get(
                    "original_name"
                ),
                "node_level": node.get(
                    "node_level"
                ),
                "taxonomy_id": node.get(
                    "taxonomy_id"
                ),
                "taxonomy_name": taxonomy.get(
                    "name"
                ),
            }
        )

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    for interaction in interactions:

        node_from = get_id(
            interaction.get("node_from")
        )

        node_to = get_id(
            interaction.get("node_to")
        )

        all_interactions.append(
            {
                "network_id": network_id,
                "interaction_id": interaction.get(
                    "id"
                ),
                "node_from": node_from,
                "node_to": node_to,
                "date": interaction.get(
                    "date"
                ),
                "direction": interaction.get(
                    "direction"
                ),
                "type": interaction.get(
                    "type"
                ),
                "method": interaction.get(
                    "method"
                ),
                "attr_id": interaction.get(
                    "attr_id"
                ),
                "value": interaction.get(
                    "value"
                ),
            }
        )

    print(
        f"Nodes: {len(nodes)}"
    )

    print(
        f"Interactions: {len(interactions)}"
    )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

nodes_df = pd.DataFrame(all_nodes)

interactions_df = pd.DataFrame(
    all_interactions
)

networks_df = pd.DataFrame(
    network_metadata
)


# ============================================================
# SAVE
# ============================================================

nodes_path = (
    PROCESSED_DIR
    / "mangal_nodes.csv"
)

interactions_path = (
    PROCESSED_DIR
    / "mangal_interactions.csv"
)

networks_path = (
    PROCESSED_DIR
    / "mangal_networks.csv"
)


nodes_df.to_csv(
    nodes_path,
    index=False
)

interactions_df.to_csv(
    interactions_path,
    index=False
)

networks_df.to_csv(
    networks_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CAMT-GNN | MANGAL MERGE COMPLETE")
print("=" * 70)

print(
    f"\nTotal nodes: "
    f"{len(nodes_df)}"
)

print(
    f"Total interactions: "
    f"{len(interactions_df)}"
)

print(
    f"Total networks: "
    f"{len(networks_df)}"
)

print("\nFiles created:")

print(nodes_path)
print(interactions_path)
print(networks_path)

print("\n✅ Day 4 Step 1 completed.")