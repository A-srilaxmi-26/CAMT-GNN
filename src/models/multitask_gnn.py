import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MultiTaskGNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()

        # Shared GNN encoder
        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        # Task 1: link prediction
        self.link_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

        # Task 2: interaction-type classification
        self.type_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def encode(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x

    def edge_features(self, z, edge_index):

        source = z[edge_index[0]]
        target = z[edge_index[1]]

        return torch.cat(
            [source, target],
            dim=1
        )

    def predict_link(self, z, edge_index):

        edge_features = self.edge_features(
            z,
            edge_index
        )

        return self.link_head(
            edge_features
        ).squeeze(-1)

    def predict_type(self, z, edge_index):

        edge_features = self.edge_features(
            z,
            edge_index
        )

        return self.type_head(
            edge_features
        )


if __name__ == "__main__":
    print("Multi-Task GNN model ready.")