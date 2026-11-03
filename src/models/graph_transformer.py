import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GraphTransformer(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64, heads=4):
        super().__init__()

        self.conv1 = TransformerConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=0.2
        )

        self.conv2 = TransformerConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=0.2
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return x


if __name__ == "__main__":
    print("Graph Transformer model ready.")