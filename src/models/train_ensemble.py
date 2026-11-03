import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from torch_geometric.utils import negative_sampling

from ensemble import EnsembleGNN


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "pyg"


def train_and_evaluate(network_id, epochs=100):

    device = torch.device("cpu")

    data = torch.load(
        DATA_DIR / f"network_{network_id}.pt",
        weights_only=False
    ).to(device)

    num_edges = data.edge_index.shape[1]

    # -----------------------------
    # Train/test split
    # -----------------------------
    perm = torch.randperm(num_edges)

    train_size = int(0.8 * num_edges)

    train_edges = data.edge_index[:, perm[:train_size]]
    test_edges = data.edge_index[:, perm[train_size:]]

    # -----------------------------
    # Model
    # -----------------------------
    model = EnsembleGNN(
        input_dim=data.x.shape[1],
        hidden_dim=64
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    # -----------------------------
    # Training
    # -----------------------------
    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        z = model(data.x, train_edges)

        # Positive edges
        pos_src = train_edges[0]
        pos_dst = train_edges[1]

        positive_score = (
            z[pos_src] * z[pos_dst]
        ).sum(dim=1)

        # Negative edges
        negative_edges = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.shape[1]
        )

        neg_src = negative_edges[0]
        neg_dst = negative_edges[1]

        negative_score = (
            z[neg_src] * z[neg_dst]
        ).sum(dim=1)

        positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_score,
            torch.ones_like(positive_score)
        )

        negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            negative_score,
            torch.zeros_like(negative_score)
        )

        loss = positive_loss + negative_loss

        loss.backward()
        optimizer.step()

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()

    with torch.no_grad():

        z = model(data.x, train_edges)

        # Positive test edges
        test_src = test_edges[0]
        test_dst = test_edges[1]

        positive_score = (
            z[test_src] * z[test_dst]
        ).sum(dim=1)

        # Negative test edges
        negative_edges = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=test_edges.shape[1]
        )

        neg_src = negative_edges[0]
        neg_dst = negative_edges[1]

        negative_score = (
            z[neg_src] * z[neg_dst]
        ).sum(dim=1)

        scores = torch.cat([
            positive_score,
            negative_score
        ]).cpu().numpy()

        labels = np.concatenate([
            np.ones(len(positive_score)),
            np.zeros(len(negative_score))
        ])

        predictions = (scores >= 0).astype(int)

        roc_auc = roc_auc_score(labels, scores)
        average_precision = average_precision_score(labels, scores)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

    print(f"\nNetwork {network_id}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":

    networks = [
        2472,
        2474,
        5027,
        5028,
        5029
    ]

    for network_id in networks:
        train_and_evaluate(network_id)