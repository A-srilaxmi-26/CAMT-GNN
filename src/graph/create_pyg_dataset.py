from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[2]

nodes_df = pd.read_csv(ROOT / "data/processed/food_web_nodes.csv")
edges_df = pd.read_csv(ROOT / "data/processed/food_web_edges.csv")

output_dir = ROOT / "data/processed/pyg"
output_dir.mkdir(parents=True, exist_ok=True)


for network_id in nodes_df["network_id"].unique():

    nodes = nodes_df[nodes_df["network_id"] == network_id].copy()
    edges = edges_df[edges_df["network_id"] == network_id].copy()

    node_ids = nodes["node_id"].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    source = edges["node_from"].map(node_to_idx)
    target = edges["node_to"].map(node_to_idx)

    valid = source.notna() & target.notna()

    edge_index = torch.tensor(
        [
            source[valid].astype(int).tolist(),
            target[valid].astype(int).tolist()
        ],
        dtype=torch.long
    )

    x = torch.eye(len(nodes), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index
    )

    torch.save(data, output_dir / f"network_{network_id}.pt")

    print(
        f"Network {network_id}: "
        f"nodes={data.num_nodes}, "
        f"edges={data.num_edges}"
    )

print("\nPyG dataset creation complete.")