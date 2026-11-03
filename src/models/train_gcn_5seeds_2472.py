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


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pyg"
    / "network_2472.pt"
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


# ============================================================
# CONFIGURATION
# ============================================================

NETWORK_ID = 2472

SEEDS = [
    42,
    43,
    44,
    45,
    46
]

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
# GCN MODEL
# ============================================================

class GCN(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.link_predictor = torch.nn.Sequential(
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


    def forward(
        self,
        x,
        edge_index
    ):

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

        return self.link_predictor(
            edge_features
        ).squeeze(-1)


# ============================================================
# EDGE SPLIT
# ============================================================

def split_edges(
    edge_index,
    test_ratio=0.20
):

    num_edges = edge_index.size(1)

    permutation = torch.randperm(
        num_edges
    )

    test_size = int(
        num_edges * test_ratio
    )

    test_indices = permutation[
        :test_size
    ]

    train_indices = permutation[
        test_size:
    ]

    train_edges = edge_index[
        :,
        train_indices
    ]

    test_edges = edge_index[
        :,
        test_indices
    ]

    return (
        train_edges,
        test_edges
    )


# ============================================================
# CREATE POSITIVE + NEGATIVE DATA
# ============================================================

def create_link_data(
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

    return (
        logits,
        labels
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CAMT-GNN | GCN | FAIR 5-SEED BASELINE")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    print("\nERROR: Dataset not found:")
    print(DATA_FILE)

    raise SystemExit(1)


print("\nLoading dataset:")
print(DATA_FILE)


data = torch.load(
    DATA_FILE,
    weights_only=False
)


print("\nNetwork:", NETWORK_ID)
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Features:", data.num_node_features)


# ============================================================
# STORE RESULTS
# ============================================================

results = []


# ============================================================
# FIVE-SEED EXPERIMENT
# ============================================================

for seed in SEEDS:

    print("\n" + "-" * 70)
    print(f"SEED: {seed}")
    print("-" * 70)

    set_seed(seed)


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_edges, test_edges = split_edges(
        data.edge_index,
        test_ratio=0.20
    )


    print(
        "Training positive edges:",
        train_edges.size(1)
    )

    print(
        "Testing positive edges:",
        test_edges.size(1)
    )


    # --------------------------------------------------------
    # NEGATIVE EDGES
    # --------------------------------------------------------

    train_negative_edges = negative_sampling(
        edge_index=train_edges,
        num_nodes=data.num_nodes,
        num_neg_samples=train_edges.size(1),
        method="sparse"
    )


    test_negative_edges = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=test_edges.size(1),
        method="sparse"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = GCN(
        input_dim=data.num_node_features,
        hidden_dim=HIDDEN_DIM
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    criterion = torch.nn.BCEWithLogitsLoss()


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        optimizer.zero_grad()


        # IMPORTANT:
        # Only training edges are used for message passing.
        z = model(
            data.x,
            train_edges
        )


        train_logits, train_labels = create_link_data(
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

        z = model(
            data.x,
            train_edges
        )


        test_logits, test_labels = create_link_data(
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


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.4f}"
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


    # --------------------------------------------------------
    # SAVE RESULT IN MEMORY
    # --------------------------------------------------------

    results.append(
        {
            "network_id": NETWORK_ID,
            "model": "GCN_5SeedFair",
            "seed": seed,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

raw_file = (
    RESULTS_DIR
    / "gcn_2472_results_5seeds.csv"
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


summary_rows = []


for metric in metrics:

    summary_rows.append(
        {
            "network_id": NETWORK_ID,
            "model": "GCN_5SeedFair",
            "metric": metric,
            "mean": results_df[metric].mean(),
            "std": results_df[metric].std(
                ddof=1
            )
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_file = (
    RESULTS_DIR
    / "gcn_2472_results_5seeds_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GCN FIVE-SEED EXPERIMENT COMPLETE")
print("=" * 70)


print("\nMean ± Std:")


for metric in metrics:

    mean_value = (
        results_df[metric].mean()
    )

    std_value = (
        results_df[metric].std(
            ddof=1
        )
    )

    print(
        f"{metric}: "
        f"{mean_value:.4f} ± "
        f"{std_value:.4f}"
    )


print("\nRaw results saved:")
print(raw_file)


print("\nSummary saved:")
print(summary_file)


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)