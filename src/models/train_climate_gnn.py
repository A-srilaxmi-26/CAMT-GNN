from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch_geometric.utils import negative_sampling

from climate_gnn import ClimateAwareGNN


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pyg_climate"
    / "network_2472_climate.pt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 43, 44, 45, 46]

HIDDEN_DIM = 64
LEARNING_RATE = 0.01
EPOCHS = 100

NETWORK_ID = 2472


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# EDGE SPLIT
# ============================================================

def split_edges(edge_index, test_ratio=0.20):

    num_edges = edge_index.size(1)

    permutation = torch.randperm(
        num_edges
    )

    test_size = int(
        num_edges * test_ratio
    )

    test_idx = permutation[:test_size]

    train_idx = permutation[test_size:]

    train_edges = edge_index[:, train_idx]

    test_edges = edge_index[:, test_idx]

    return train_edges, test_edges


# ============================================================
# LINK SCORE
# ============================================================

def make_link_labels(
    model,
    z,
    positive_edges,
    negative_edges
):

    positive_logits = model.predict_link(
        z,
        positive_edges
    )

    negative_logits = model.predict_link(
        z,
        negative_edges
    )

    logits = torch.cat(
        [
            positive_logits,
            negative_logits
        ]
    )

    labels = torch.cat(
        [
            torch.ones(
                positive_logits.size(0)
            ),
            torch.zeros(
                negative_logits.size(0)
            )
        ]
    )

    return logits, labels


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("=" * 70)
print("CAMT-GNN | CLIMATE-AWARE GNN")
print("LINK PREDICTION EXPERIMENT")
print("=" * 70)


if not DATA_FILE.exists():

    print("\nERROR: Climate PyG dataset not found:")
    print(DATA_FILE)

    raise SystemExit(1)


print("\nLoading:")
print(DATA_FILE)


data = torch.load(
    DATA_FILE,
    weights_only=False
)


print("\nNetwork ID:", data.network_id)
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Features:", data.num_node_features)

print("\nClimate:")
print("SST mean:", data.sst_mean)
print("SST min :", data.sst_min)
print("SST max :", data.sst_max)
print("SST std :", data.sst_std)


# ============================================================
# RUN FIVE SEEDS
# ============================================================

results = []


for seed in SEEDS:

    print("\n" + "-" * 70)
    print("SEED:", seed)
    print("-" * 70)

    set_seed(seed)

    train_edges, test_edges = split_edges(
        data.edge_index
    )


    # --------------------------------------------------------
    # NEGATIVE TRAINING EDGES
    # --------------------------------------------------------

    train_negative_edges = negative_sampling(
        edge_index=train_edges,
        num_nodes=data.num_nodes,
        num_neg_samples=train_edges.size(1),
        method="sparse"
    )


    # --------------------------------------------------------
    # NEGATIVE TEST EDGES
    # --------------------------------------------------------

    test_negative_edges = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=test_edges.size(1),
        method="sparse"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = ClimateAwareGNN(
        input_dim=data.num_node_features,
        climate_dim=4,
        hidden_dim=HIDDEN_DIM
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    criterion = torch.nn.BCEWithLogitsLoss()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        optimizer.zero_grad()


        # IMPORTANT:
        # Only training edges are used for message passing.
        z = model.encode(
            data.x,
            train_edges,
            data.x[:, -4:]
        )


        train_logits, train_labels = make_link_labels(
            model,
            z,
            train_edges,
            train_negative_edges
        )


        loss = criterion(
            train_logits,
            train_labels
        )


        loss.backward()

        optimizer.step()


        if epoch % 20 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f}"
            )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        z = model.encode(
            data.x,
            train_edges,
            data.x[:, -4:]
        )

        test_logits, test_labels = make_link_labels(
            model,
            z,
            test_edges,
            test_negative_edges
        )

        probabilities = torch.sigmoid(
            test_logits
        ).cpu().numpy()

        labels = test_labels.cpu().numpy()


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    average_precision = average_precision_score(
        labels,
        probabilities
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )


    print("\nResults:")
    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )


    results.append(
        {
            "network_id": NETWORK_ID,
            "model": "ClimateAwareGNN",
            "seed": seed,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLIMATE-AWARE GNN EXPERIMENT COMPLETE")
print("=" * 70)

print("\nFive-seed results:")

for result in results:

    print(
        f"Seed {result['seed']}: "
        f"ROC-AUC={result['roc_auc']:.4f}, "
        f"AP={result['average_precision']:.4f}, "
        f"F1={result['f1']:.4f}"
    )


print("\nMean ± Std:")

for metric in [
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1"
]:

    values = [
        result[metric]
        for result in results
    ]

    print(
        f"{metric}: "
        f"{np.mean(values):.4f} ± "
        f"{np.std(values, ddof=1):.4f}"
    )