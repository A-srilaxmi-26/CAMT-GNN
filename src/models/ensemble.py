import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv


class EnsembleGNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # GCN
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # GAT
        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=4,
            dropout=0.2
        )
        self.gat2 = GATConv(
            hidden_dim * 4,
            hidden_dim,
            heads=1,
            dropout=0.2
        )

        # Graph Transformer
        self.trans1 = TransformerConv(
            input_dim,
            hidden_dim,
            heads=4,
            dropout=0.2
        )
        self.trans2 = TransformerConv(
            hidden_dim * 4,
            hidden_dim,
            heads=1,
            dropout=0.2
        )

    def forward(self, x, edge_index):

        # GCN embedding
        gcn = self.gcn1(x, edge_index)
        gcn = F.relu(gcn)
        gcn = self.gcn2(gcn, edge_index)

        # GAT embedding
        gat = self.gat1(x, edge_index)
        gat = F.relu(gat)
        gat = self.gat2(gat, edge_index)

        # Transformer embedding
        trans = self.trans1(x, edge_index)
        trans = F.relu(trans)
        trans = self.trans2(trans, edge_index)

        # Ensemble
        z = (gcn + gat + trans) / 3.0

        return z


if __name__ == "__main__":
    print("Ensemble GNN model ready.")