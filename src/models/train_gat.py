from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parents[2]


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)


def run(network_id, epochs=100):

    data = torch.load(
        ROOT / f"data/processed/pyg/network_{network_id}.pt",
        weights_only=False
    )

    num_edges = data.edge_index.size(1)
    split = int(0.8 * num_edges)

    perm = torch.randperm(num_edges)
    train_edges = data.edge_index[:, perm[:split]]
    test_edges = data.edge_index[:, perm[split:]]

    model = GAT(data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):

        model.train()
        optimizer.zero_grad()

        z = model(data.x, train_edges)

        neg_edges = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.size(1)
        )

        pos_score = (z[train_edges[0]] * z[train_edges[1]]).sum(dim=1)
        neg_score = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])

        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():

        z = model(data.x, train_edges)

        neg_test = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=test_edges.size(1)
        )

        pos_score = (z[test_edges[0]] * z[test_edges[1]]).sum(dim=1)
        neg_score = (z[neg_test[0]] * z[neg_test[1]]).sum(dim=1)

        scores = torch.sigmoid(
            torch.cat([pos_score, neg_score])
        ).numpy()

        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ]).numpy()

    predictions = (scores >= 0.5).astype(int)

    print(f"\nNetwork {network_id}")
    print(f"ROC-AUC: {roc_auc_score(labels, scores):.4f}")
    print(f"Average Precision: {average_precision_score(labels, scores):.4f}")
    print(f"Precision: {precision_score(labels, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(labels, predictions, zero_division=0):.4f}")
    print(f"F1: {f1_score(labels, predictions, zero_division=0):.4f}")


if __name__ == "__main__":
    for network_id in [2472, 2474, 5027, 5028, 5029]:
        run(network_id)