import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TemporalGNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.temporal = torch.nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, snapshots):

        embeddings = []

        for data in snapshots:

            x = self.gcn1(data.x, data.edge_index)
            x = F.relu(x)

            x = self.gcn2(x, data.edge_index)
            x = F.relu(x)

            embeddings.append(x)

        x = torch.stack(embeddings, dim=1)

        x, _ = self.temporal(x)

        return x[:, -1, :]


if __name__ == "__main__":
    print("Temporal GNN model ready.")