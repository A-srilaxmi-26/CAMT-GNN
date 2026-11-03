from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]

edges = pd.read_csv(
    ROOT / "data/processed/temporal_food_web_edges.csv"
)

nodes = pd.read_csv(
    ROOT / "data/processed/food_web_nodes.csv"
)

output_dir = ROOT / "data/processed/temporal_pyg"
output_dir.mkdir(parents=True, exist_ok=True)

for network_id in [2472, 2474]:

    network_nodes = nodes[nodes["network_id"] == network_id].copy()
    network_edges = edges[edges["network_id"] == network_id].copy()

    node_ids = network_nodes["node_id"].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    for date in sorted(network_edges["date"].unique()):

        snapshot = network_edges[
            network_edges["date"] == date
        ]

        source = snapshot["node_from"].map(node_to_idx)
        target = snapshot["node_to"].map(node_to_idx)

        valid = source.notna() & target.notna()

        edge_index = torch.tensor(
            [
                source[valid].astype(int).tolist(),
                target[valid].astype(int).tolist()
            ],
            dtype=torch.long
        )

        x = torch.eye(len(network_nodes), dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index
        )

        year = pd.to_datetime(date).year

        torch.save(
            data,
            output_dir / f"network_{network_id}_{year}.pt"
        )

        print(
            f"Network {network_id} | "
            f"{year} | "
            f"nodes={data.num_nodes} | "
            f"edges={data.num_edges}"
        )

print("\nTemporal graph construction complete.")