from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "food_web_nodes.csv"
)

EDGES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "food_web_edges.csv"
)

CLIMATE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "mangal_climate_networks.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pyg_climate"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("CAMT-GNN | CLIMATE-AWARE PyG DATASET")
print("=" * 70)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

nodes_df = pd.read_csv(NODES_FILE)
edges_df = pd.read_csv(EDGES_FILE)
climate_df = pd.read_csv(CLIMATE_FILE)


# ------------------------------------------------------------
# NETWORKS WITH REAL CLIMATE DATA
# ------------------------------------------------------------

climate_df = climate_df[
    climate_df["sst_mean"].notna()
].copy()


print("\nNetworks with real climate data:")
print(
    climate_df["network_id"].tolist()
)


# ------------------------------------------------------------
# BUILD NETWORK GRAPHS
# ------------------------------------------------------------

for network_id in climate_df["network_id"].unique():

    nodes = nodes_df[
        nodes_df["network_id"] == network_id
    ].copy()

    edges = edges_df[
        edges_df["network_id"] == network_id
    ].copy()

    climate = climate_df[
        climate_df["network_id"] == network_id
    ].iloc[0]


    # --------------------------------------------------------
    # NODE MAPPING
    # Same ordering used by the existing PyG dataset
    # --------------------------------------------------------

    node_ids = nodes["node_id"].tolist()

    node_to_idx = {
        node_id: i
        for i, node_id in enumerate(node_ids)
    }


    # --------------------------------------------------------
    # EDGE INDEX
    # --------------------------------------------------------

    source = edges["node_from"].map(node_to_idx)
    target = edges["node_to"].map(node_to_idx)

    valid = (
        source.notna()
        & target.notna()
    )

    edge_index = torch.tensor(
        [
            source[valid].astype(int).tolist(),
            target[valid].astype(int).tolist()
        ],
        dtype=torch.long
    )


    # --------------------------------------------------------
    # BASE NODE FEATURES
    # --------------------------------------------------------

    x_base = torch.eye(
        len(nodes),
        dtype=torch.float
    )


    # --------------------------------------------------------
    # REAL CLIMATE FEATURES
    #
    # Same network-level climate context is attached
    # to every node.
    # --------------------------------------------------------

    climate_features = torch.tensor(
        [
            [
                float(climate["sst_mean"]),
                float(climate["sst_min"]),
                float(climate["sst_max"]),
                float(climate["sst_std"])
            ]
        ],
        dtype=torch.float
    )


    climate_features = climate_features.repeat(
        len(nodes),
        1
    )


    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------

    x = torch.cat(
        [
            x_base,
            climate_features
        ],
        dim=1
    )


    # --------------------------------------------------------
    # CREATE PyG DATA
    # --------------------------------------------------------

    data = Data(
        x=x,
        edge_index=edge_index
    )


    data.network_id = int(network_id)

    data.sst_mean = float(
        climate["sst_mean"]
    )

    data.sst_min = float(
        climate["sst_min"]
    )

    data.sst_max = float(
        climate["sst_max"]
    )

    data.sst_std = float(
        climate["sst_std"]
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"network_{network_id}_climate.pt"
    )

    torch.save(
        data,
        output_file
    )


    print(
        f"\nNetwork {network_id}: "
        f"nodes={data.num_nodes}, "
        f"edges={data.num_edges}, "
        f"features={data.num_node_features}"
    )

    print(
        f"SST mean={data.sst_mean:.6f}"
    )


print("\n" + "=" * 70)
print("SUCCESS — CLIMATE-AWARE PyG DATASET CREATED")
print("=" * 70)