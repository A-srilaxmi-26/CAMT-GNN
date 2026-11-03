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

# ============================================================
# CONFIG
# ============================================================

BASE_GRAPH = "data/processed/pyg/network_2472.pt"
CLIMATE_CSV = "data/processed/network_2472_climate_spatial_features.csv"

RESULTS_RAW = "data/results/spatial_climate_ablation_2472_5seeds.csv"
RESULTS_SUMMARY = "data/results/spatial_climate_ablation_2472_summary.csv"

SEEDS = [42, 43, 44, 45, 46]

EPOCHS = 100
LR = 0.01
HIDDEN = 64
TEST_RATIO = 0.20

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# GCN MODEL
# ============================================================

class GCNLinkPredictor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = GCNConv(input_dim, HIDDEN)
        self.conv2 = GCNConv(HIDDEN, HIDDEN)

        self.predictor = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        return x

    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]

        pair = torch.cat([src, dst], dim=1)

        return self.predictor(pair).view(-1)


# ============================================================
# EDGE SPLIT
# ============================================================

def split_edges(edge_index, seed):

    set_seed(seed)

    num_edges = edge_index.shape[1]

    permutation = torch.randperm(num_edges)

    test_size = int(num_edges * TEST_RATIO)

    test_idx = permutation[:test_size]
    train_idx = permutation[test_size:]

    train_edges = edge_index[:, train_idx]
    test_edges = edge_index[:, test_idx]

    return train_edges, test_edges


# ============================================================
# NEGATIVE SAMPLING
# ============================================================

def negative_sampling(num_nodes, positive_edges, num_samples, seed):

    set_seed(seed)

    positive_set = set(
        (int(positive_edges[0, i]), int(positive_edges[1, i]))
        for i in range(positive_edges.shape[1])
    )

    negatives = []

    while len(negatives) < num_samples:

        src = random.randrange(num_nodes)
        dst = random.randrange(num_nodes)

        if src == dst:
            continue

        if (src, dst) in positive_set:
            continue

        negatives.append((src, dst))

    return torch.tensor(
        negatives,
        dtype=torch.long
    ).t().contiguous()


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def run_condition(
    x,
    train_edges,
    train_neg_edges,
    test_edges,
    test_neg_edges,
    seed
):

    set_seed(seed)

    model = GCNLinkPredictor(
        input_dim=x.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        z = model.encode(
            x,
            train_edges
        )

        positive_logits = model.decode(
            z,
            train_edges
        )

        negative_logits = model.decode(
            z,
            train_neg_edges
        )

        logits = torch.cat([
            positive_logits,
            negative_logits
        ])

        labels = torch.cat([
            torch.ones_like(positive_logits),
            torch.zeros_like(negative_logits)
        ])

        loss = criterion(
            logits,
            labels
        )

        loss.backward()
        optimizer.step()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        z = model.encode(
            x,
            train_edges
        )

        positive_scores = torch.sigmoid(
            model.decode(z, test_edges)
        )

        negative_scores = torch.sigmoid(
            model.decode(z, test_neg_edges)
        )

    scores = torch.cat([
        positive_scores,
        negative_scores
    ]).cpu().numpy()

    labels = np.concatenate([
        np.ones(len(positive_scores)),
        np.zeros(len(negative_scores))
    ])

    predictions = (scores >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
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


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CAMT-GNN | SPATIAL CLIMATE ABLATION | NETWORK 2472")
print("=" * 70)

# ------------------------------------------------------------
# Load base graph
# ------------------------------------------------------------

base = torch.load(
    BASE_GRAPH,
    weights_only=False
)

base_x = base.x.float()
edge_index = base.edge_index.long()

print("\nBase graph:")
print("Nodes:", base_x.shape[0])
print("Base features:", base_x.shape)
print("Edges:", edge_index.shape[1])

# ------------------------------------------------------------
# Load real spatial SST features
# ------------------------------------------------------------

climate_df = pd.read_csv(CLIMATE_CSV)

climate_columns = [
    "sst_mean",
    "sst_min",
    "sst_max",
    "sst_std",
    "sst_range",
    "sst_lat_gradient_mean",
    "sst_lon_gradient_mean",
]

climate_values = climate_df[climate_columns].iloc[0].values.astype(
    np.float32
)

climate_features = torch.tensor(
    climate_values,
    dtype=torch.float32
)

print("\nReal spatial SST features:")

for name, value in zip(
    climate_columns,
    climate_values
):
    print(f"{name}: {value:.6f}")

# ------------------------------------------------------------
# Repeat network-level climate context for every node
# ------------------------------------------------------------

climate_node_features = climate_features.repeat(
    base_x.shape[0],
    1
)

with_climate_x = torch.cat(
    [
        base_x,
        climate_node_features
    ],
    dim=1
)

print("\nFeature shapes:")
print("No climate :", base_x.shape)
print("With climate:", with_climate_x.shape)

assert with_climate_x.shape[1] == base_x.shape[1] + 7

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

results = []

# ============================================================
# SAME SPLIT FOR BOTH CONDITIONS
# ============================================================

for seed in SEEDS:

    print("\n" + "-" * 70)
    print(f"SEED: {seed}")
    print("-" * 70)

    # --------------------------------------------------------
    # Same positive edge split
    # --------------------------------------------------------

    train_edges, test_edges = split_edges(
        edge_index,
        seed
    )

    # --------------------------------------------------------
    # Same negative samples for both models
    # --------------------------------------------------------

    train_neg_edges = negative_sampling(
        base_x.shape[0],
        train_edges,
        train_edges.shape[1],
        seed + 1000
    )

    test_neg_edges = negative_sampling(
        base_x.shape[0],
        edge_index,
        test_edges.shape[1],
        seed + 2000
    )

    # --------------------------------------------------------
    # NO CLIMATE
    # --------------------------------------------------------

    no_climate = run_condition(
        base_x,
        train_edges,
        train_neg_edges,
        test_edges,
        test_neg_edges,
        seed
    )

    print(
        f"No climate    | "
        f"ROC-AUC={no_climate['roc_auc']:.4f} | "
        f"F1={no_climate['f1']:.4f}"
    )

    results.append({
        "seed": seed,
        "condition": "GCN_NoClimate",
        **no_climate
    })

    # --------------------------------------------------------
    # SPATIAL CLIMATE
    # --------------------------------------------------------

    spatial_climate = run_condition(
        with_climate_x,
        train_edges,
        train_neg_edges,
        test_edges,
        test_neg_edges,
        seed
    )

    print(
        f"Spatial climate | "
        f"ROC-AUC={spatial_climate['roc_auc']:.4f} | "
        f"F1={spatial_climate['f1']:.4f}"
    )

    results.append({
        "seed": seed,
        "condition": "GCN_SpatialClimate",
        **spatial_climate
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

os.makedirs(
    "data/results",
    exist_ok=True
)

results_df.to_csv(
    RESULTS_RAW,
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
    RESULTS_SUMMARY
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL CLIMATE ABLATION COMPLETE")
print("=" * 70)

for condition in summary.index:

    print(f"\n{condition}")

    for metric in [
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1"
    ]:

        mean = summary.loc[
            condition,
            (metric, "mean")
        ]

        std = summary.loc[
            condition,
            (metric, "std")
        ]

        print(
            f"{metric}: "
            f"{mean:.4f} ± {std:.4f}"
        )

print("\nRaw results:")
print(RESULTS_RAW)

print("\nSummary:")
print(RESULTS_SUMMARY)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)