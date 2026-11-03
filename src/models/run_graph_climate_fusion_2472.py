import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_GRAPH = "data/processed/pyg/network_2472.pt"
CLIMATE_CSV = "data/processed/network_2472_climate_spatial_features.csv"

RAW_OUT = "data/results/graph_climate_fusion_2472_5seeds.csv"
SUMMARY_OUT = "data/results/graph_climate_fusion_2472_summary.csv"

SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 100
LR = 0.01
HIDDEN = 64
TEST_RATIO = 0.20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GraphClimateGCN(nn.Module):

    def __init__(self, input_dim, climate_dim=7):
        super().__init__()

        self.conv1 = GCNConv(input_dim, HIDDEN)
        self.conv2 = GCNConv(HIDDEN, HIDDEN)

        # Graph-level climate encoder
        self.climate_encoder = nn.Sequential(
            nn.Linear(climate_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Fuse graph representation + climate representation
        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN + 16, HIDDEN),
            nn.ReLU()
        )

        self.predictor = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )

    def encode(self, x, message_edges, climate):

        h = self.conv1(x, message_edges)
        h = torch.relu(h)

        h = self.conv2(h, message_edges)
        h = torch.relu(h)

        climate_embedding = self.climate_encoder(
            climate
        )

        # Same graph-level climate context is fused
        # with every node embedding after GCN encoding.
        climate_embedding = climate_embedding.unsqueeze(0)

        climate_embedding = climate_embedding.repeat(
            h.shape[0],
            1
        )

        fused = torch.cat(
            [h, climate_embedding],
            dim=1
        )

        return self.fusion(fused)

    def decode(self, z, edges):

        src = z[edges[0]]
        dst = z[edges[1]]

        pair = torch.cat(
            [src, dst],
            dim=1
        )

        return self.predictor(pair).view(-1)


def split_edges(edge_index, seed):

    set_seed(seed)

    n = edge_index.shape[1]

    perm = torch.randperm(n)

    test_size = int(n * TEST_RATIO)

    test_idx = perm[:test_size]
    train_idx = perm[test_size:]

    return (
        edge_index[:, train_idx],
        edge_index[:, test_idx]
    )


def negative_sampling(
    num_nodes,
    positive_edges,
    num_samples,
    seed
):

    set_seed(seed)

    positive = set(
        (
            int(positive_edges[0, i]),
            int(positive_edges[1, i])
        )
        for i in range(positive_edges.shape[1])
    )

    negatives = []

    while len(negatives) < num_samples:

        src = random.randrange(num_nodes)
        dst = random.randrange(num_nodes)

        if src == dst:
            continue

        if (src, dst) in positive:
            continue

        negatives.append((src, dst))

    return torch.tensor(
        negatives,
        dtype=torch.long
    ).t().contiguous()


def train_and_evaluate(
    x,
    climate,
    train_edges,
    train_neg,
    test_edges,
    test_neg,
    seed
):

    set_seed(seed)

    model = GraphClimateGCN(
        input_dim=x.shape[1],
        climate_dim=7
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        z = model.encode(
            x,
            train_edges,
            climate
        )

        pos_logits = model.decode(
            z,
            train_edges
        )

        neg_logits = model.decode(
            z,
            train_neg
        )

        logits = torch.cat(
            [pos_logits, neg_logits]
        )

        labels = torch.cat(
            [
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits)
            ]
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():

        z = model.encode(
            x,
            train_edges,
            climate
        )

        pos_scores = torch.sigmoid(
            model.decode(z, test_edges)
        )

        neg_scores = torch.sigmoid(
            model.decode(z, test_neg)
        )

    scores = torch.cat(
        [pos_scores, neg_scores]
    ).cpu().numpy()

    labels = np.concatenate(
        [
            np.ones(len(pos_scores)),
            np.zeros(len(neg_scores))
        ]
    )

    predictions = (
        scores >= 0.5
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            labels,
            scores
        ),
        "average_precision": average_precision_score(
            labels,
            scores
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        )
    }


print("=" * 70)
print("CAMT-GNN | GRAPH-LEVEL CLIMATE FUSION | NETWORK 2472")
print("=" * 70)

# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------

data = torch.load(
    BASE_GRAPH,
    weights_only=False
)

x = data.x.float()
edge_index = data.edge_index.long()

print("\nGraph:")
print("Nodes:", x.shape[0])
print("Features:", x.shape)
print("Edges:", edge_index.shape[1])

# ------------------------------------------------------------
# Load real climate features
# ------------------------------------------------------------

climate_df = pd.read_csv(
    CLIMATE_CSV
)

climate_columns = [
    "sst_mean",
    "sst_min",
    "sst_max",
    "sst_std",
    "sst_range",
    "sst_lat_gradient_mean",
    "sst_lon_gradient_mean",
]

climate = torch.tensor(
    climate_df[climate_columns]
    .iloc[0]
    .values
    .astype(np.float32)
)

print("\nGraph-level climate context:")

for name, value in zip(
    climate_columns,
    climate.numpy()
):
    print(
        f"{name}: {value:.6f}"
    )

print("\nClimate shape:", climate.shape)

# ------------------------------------------------------------
# Run five seeds
# ------------------------------------------------------------

results = []

for seed in SEEDS:

    print("\n" + "-" * 70)
    print(f"SEED: {seed}")
    print("-" * 70)

    train_edges, test_edges = split_edges(
        edge_index,
        seed
    )

    train_neg = negative_sampling(
        x.shape[0],
        train_edges,
        train_edges.shape[1],
        seed + 1000
    )

    test_neg = negative_sampling(
        x.shape[0],
        edge_index,
        test_edges.shape[1],
        seed + 2000
    )

    metrics = train_and_evaluate(
        x,
        climate,
        train_edges,
        train_neg,
        test_edges,
        test_neg,
        seed
    )

    print(
        f"Graph climate | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )

    results.append({
        "seed": seed,
        "condition": "GCN_GraphClimateFusion",
        **metrics
    })


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs(
    "data/results",
    exist_ok=True
)

results_df.to_csv(
    RAW_OUT,
    index=False
)

summary = (
    results_df
    .groupby("condition")
    [
        [
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1"
        ]
    ]
    .agg(["mean", "std"])
)

summary.to_csv(
    SUMMARY_OUT
)

print("\n" + "=" * 70)
print("GRAPH CLIMATE FUSION COMPLETE")
print("=" * 70)

for metric in [
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1"
]:

    mean = summary.loc[
        "GCN_GraphClimateFusion",
        (metric, "mean")
    ]

    std = summary.loc[
        "GCN_GraphClimateFusion",
        (metric, "std")
    ]

    print(
        f"{metric}: "
        f"{mean:.4f} ± {std:.4f}"
    )

print("\nRaw results:")
print(RAW_OUT)

print("\nSummary:")
print(SUMMARY_OUT)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)