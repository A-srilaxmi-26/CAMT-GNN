from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling


ROOT = Path(__file__).resolve().parents[2]

BASE_DATA = (
    ROOT
    / "data"
    / "processed"
    / "pyg"
    / "network_2472.pt"
)

CLIMATE_DATA = (
    ROOT
    / "data"
    / "processed"
    / "pyg_climate"
    / "network_2472_climate.pt"
)

RESULTS_DIR = (
    ROOT
    / "data"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


NETWORK_ID = 2472

SEEDS = [42, 43, 44, 45, 46]

HIDDEN_DIM = 64
LEARNING_RATE = 0.01
EPOCHS = 100


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# GCN
# ============================================================

class GCN(torch.nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            HIDDEN_DIM
        )

        self.conv2 = GCNConv(
            HIDDEN_DIM,
            HIDDEN_DIM
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                HIDDEN_DIM * 2,
                HIDDEN_DIM
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                HIDDEN_DIM,
                1
            )
        )


    def encode(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


    def predict(self, z, edge_index):

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

        return self.predictor(
            edge_features
        ).squeeze(-1)


# ============================================================
# EDGE SPLIT
# ============================================================

def split_edges(edge_index):

    num_edges = edge_index.size(1)

    permutation = torch.randperm(
        num_edges
    )

    test_size = int(
        num_edges * 0.20
    )

    test_idx = permutation[:test_size]

    train_idx = permutation[test_size:]

    return (
        edge_index[:, train_idx],
        edge_index[:, test_idx]
    )


# ============================================================
# EVALUATION DATA
# ============================================================

def create_labels(
    model,
    z,
    positive_edges,
    negative_edges
):

    positive_logits = model.predict(
        z,
        positive_edges
    )

    negative_logits = model.predict(
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
# TRAIN ONE CONDITION
# ============================================================

def run_condition(
    x,
    full_edge_index,
    train_edges,
    test_edges,
    train_negative_edges,
    test_negative_edges,
    seed,
    condition
):

    set_seed(seed)

    model = GCN(
        input_dim=x.size(1)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        optimizer.zero_grad()

        z = model.encode(
            x,
            train_edges
        )

        train_logits, train_labels = create_labels(
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

    model.eval()

    with torch.no_grad():

        z = model.encode(
            x,
            train_edges
        )

        test_logits, test_labels = create_labels(
            model,
            z,
            test_edges,
            test_negative_edges
        )

        probabilities = torch.sigmoid(
            test_logits
        ).cpu().numpy()

        labels = (
            test_labels
            .cpu()
            .numpy()
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "network_id": NETWORK_ID,
        "condition": condition,
        "seed": seed,
        "roc_auc": roc_auc_score(
            labels,
            probabilities
        ),
        "average_precision": average_precision_score(
            labels,
            probabilities
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


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CAMT-GNN | CLIMATE ABLATION | NETWORK 2472")
print("=" * 70)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

base_data = torch.load(
    BASE_DATA,
    weights_only=False
)

climate_data = torch.load(
    CLIMATE_DATA,
    weights_only=False
)


print("\nBase features:", base_data.x.shape)
print("Climate features:", climate_data.x.shape)
print("Edges:", base_data.edge_index.size(1))


# ------------------------------------------------------------
# VERIFY SAME GRAPH
# ------------------------------------------------------------

if not torch.equal(
    base_data.edge_index,
    climate_data.edge_index
):

    print(
        "\nERROR: Base and climate graphs "
        "do not have identical edge_index."
    )

    raise SystemExit(1)


if base_data.num_nodes != climate_data.num_nodes:

    print(
        "\nERROR: Node counts do not match."
    )

    raise SystemExit(1)


print("\nGraph structure verified.")


# ------------------------------------------------------------
# CLIMATE FEATURES
# ------------------------------------------------------------

climate_features = climate_data.x[:, -4:]


print(
    "Climate feature shape:",
    climate_features.shape
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

results = []


# ============================================================
# SAME SPLIT FOR BOTH CONDITIONS
# ============================================================

for seed in SEEDS:

    print("\n" + "-" * 70)
    print("SEED:", seed)
    print("-" * 70)

    set_seed(seed)

    train_edges, test_edges = split_edges(
        base_data.edge_index
    )

    train_negative_edges = negative_sampling(
        edge_index=train_edges,
        num_nodes=base_data.num_nodes,
        num_neg_samples=train_edges.size(1)
    )

    test_negative_edges = negative_sampling(
        edge_index=base_data.edge_index,
        num_nodes=base_data.num_nodes,
        num_neg_samples=test_edges.size(1)
    )


    # --------------------------------------------------------
    # NO CLIMATE
    # --------------------------------------------------------

    result_base = run_condition(
        base_data.x,
        base_data.edge_index,
        train_edges,
        test_edges,
        train_negative_edges,
        test_negative_edges,
        seed,
        "GCN_NoClimate"
    )

    results.append(
        result_base
    )


    # --------------------------------------------------------
    # WITH CLIMATE
    # --------------------------------------------------------

    result_climate = run_condition(
        climate_data.x,
        climate_data.edge_index,
        train_edges,
        test_edges,
        train_negative_edges,
        test_negative_edges,
        seed,
        "GCN_WithClimate"
    )

    results.append(
        result_climate
    )


    print(
        f"No climate | "
        f"ROC-AUC={result_base['roc_auc']:.4f} | "
        f"F1={result_base['f1']:.4f}"
    )

    print(
        f"With climate | "
        f"ROC-AUC={result_climate['roc_auc']:.4f} | "
        f"F1={result_climate['f1']:.4f}"
    )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

raw_file = (
    RESULTS_DIR
    / "climate_ablation_2472_5seeds.csv"
)

results_df.to_csv(
    raw_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

metrics = [
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1"
]

summary = (
    results_df
    .groupby("condition")[metrics]
    .agg(["mean", "std"])
)

summary_file = (
    RESULTS_DIR
    / "climate_ablation_2472_summary.csv"
)

summary.to_csv(
    summary_file
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLIMATE ABLATION COMPLETE")
print("=" * 70)

print("\nMean ± Std:")

for condition in [
    "GCN_NoClimate",
    "GCN_WithClimate"
]:

    subset = results_df[
        results_df["condition"] == condition
    ]

    print("\n" + condition)

    for metric in metrics:

        print(
            f"{metric}: "
            f"{subset[metric].mean():.4f} ± "
            f"{subset[metric].std(ddof=1):.4f}"
        )


print("\nRaw results:")
print(raw_file)

print("\nSummary:")
print(summary_file)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)