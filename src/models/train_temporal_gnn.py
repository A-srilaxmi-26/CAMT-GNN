from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[2]


class TemporalGNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.gru = torch.nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, snapshots):

        embeddings = []

        for data in snapshots:
            x = F.relu(self.gcn1(data.x, data.edge_index))
            x = F.relu(self.gcn2(x, data.edge_index))
            embeddings.append(x)

        x = torch.stack(embeddings, dim=1)

        x, _ = self.gru(x)

        return x


def run_temporal_representation():

    snapshot_2005 = torch.load(
        ROOT / "data/processed/temporal_pyg/network_2472_2005.pt",
        weights_only=False
    )

    snapshot_2007 = torch.load(
        ROOT / "data/processed/temporal_pyg/network_2474_2007.pt",
        weights_only=False
    )

    snapshots = [snapshot_2005, snapshot_2007]

    model = TemporalGNN(
        input_dim=snapshot_2005.num_node_features
    )

    model.eval()

    with torch.no_grad():
        temporal_embeddings = model(snapshots)

    print("Temporal GNN representation completed.")
    print(
        f"2005 snapshot: network 2472 | "
        f"nodes={snapshot_2005.num_nodes} | "
        f"edges={snapshot_2005.num_edges}"
    )
    print(
        f"2007 snapshot: network 2474 | "
        f"nodes={snapshot_2007.num_nodes} | "
        f"edges={snapshot_2007.num_edges}"
    )
    print(
        f"Temporal embedding shape: "
        f"{temporal_embeddings.shape}"
    )


if __name__ == "__main__":
    run_temporal_representation()