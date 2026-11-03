import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ClimateAwareGNN(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        climate_dim=4,
        hidden_dim=64
    ):
        super().__init__()

        self.graph_conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.graph_conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.climate_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                climate_dim,
                16
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                16,
                16
            ),
            torch.nn.ReLU()
        )

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_dim + 16,
                hidden_dim
            ),
            torch.nn.ReLU()
        )

        self.link_head = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden_dim,
                1
            )
        )

    def encode(
        self,
        x,
        edge_index,
        climate_features
    ):

        x = self.graph_conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.graph_conv2(
            x,
            edge_index
        )

        climate = self.climate_encoder(
            climate_features
        )

        combined = torch.cat(
            [
                x,
                climate
            ],
            dim=1
        )

        z = self.fusion(combined)

        return z

    def predict_link(
        self,
        z,
        edge_index
    ):

        source = z[
            edge_index[0]
        ]

        target = z[
            edge_index[1]
        ]

        edge_features = torch.cat(
            [
                source,
                target
            ],
            dim=1
        )

        return self.link_head(
            edge_features
        ).squeeze(-1)


if __name__ == "__main__":
    print("Climate-Aware GNN model ready.")