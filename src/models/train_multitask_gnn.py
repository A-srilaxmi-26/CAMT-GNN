import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from torch_geometric.utils import negative_sampling

from multitask_gnn import MultiTaskGNN


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "pyg"
NODES_FILE = ROOT / "data" / "processed" / "food_web_nodes.csv"
EDGES_FILE = ROOT / "data" / "processed" / "multitask_edges.csv"

RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

nodes_df = pd.read_csv(NODES_FILE)
edges_df = pd.read_csv(EDGES_FILE)


# ============================================================
# GLOBAL RESULTS STORAGE
# ============================================================

RESULTS = []


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_and_evaluate(network_id, seed, epochs=100):

    set_seed(seed)

    device = torch.device("cpu")

    # --------------------------------------------------------
    # Load PyG graph
    # --------------------------------------------------------

    data = torch.load(
        DATA_DIR / f"network_{network_id}.pt",
        weights_only=False
    ).to(device)

    # --------------------------------------------------------
    # Recreate EXACT same node mapping
    # used during PyG dataset creation
    # --------------------------------------------------------

    nodes = nodes_df[
        nodes_df["network_id"] == network_id
    ].copy()

    node_ids = nodes["node_id"].tolist()

    node_to_idx = {
        node_id: i
        for i, node_id in enumerate(node_ids)
    }

    # --------------------------------------------------------
    # Get network edges
    # --------------------------------------------------------

    network_edges = edges_df[
        edges_df["network_id"] == network_id
    ].copy()

    # --------------------------------------------------------
    # Convert original Mangal node IDs
    # to local PyG node indices
    # --------------------------------------------------------

    network_edges["source_idx"] = (
        network_edges["node_from"]
        .map(node_to_idx)
    )

    network_edges["target_idx"] = (
        network_edges["node_to"]
        .map(node_to_idx)
    )

    network_edges = network_edges.dropna(
        subset=["source_idx", "target_idx"]
    ).copy()

    network_edges["source_idx"] = (
        network_edges["source_idx"].astype(int)
    )

    network_edges["target_idx"] = (
        network_edges["target_idx"].astype(int)
    )

    # --------------------------------------------------------
    # Build edge -> interaction type mapping
    # --------------------------------------------------------

    edge_type_map = {}

    for _, row in network_edges.iterrows():

        key = (
            int(row["source_idx"]),
            int(row["target_idx"])
        )

        edge_type_map[key] = row[
            "interaction_type_target"
        ]

    # --------------------------------------------------------
    # Align labels with actual PyG edges
    # --------------------------------------------------------

    graph_edges = data.edge_index.t().cpu().numpy()

    type_labels = []

    for source, target in graph_edges:

        key = (
            int(source),
            int(target)
        )

        type_labels.append(
            edge_type_map.get(key, np.nan)
        )

    type_labels = np.array(
        type_labels,
        dtype=float
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(f"Network {network_id} | Seed {seed}")
    print("-" * 60)

    print(f"Graph edges: {len(graph_edges)}")

    print(
        f"Labeled edges: "
        f"{np.sum(~np.isnan(type_labels))}"
    )

    print(
        f"Unspecified edges: "
        f"{np.sum(np.isnan(type_labels))}"
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    num_edges = data.edge_index.shape[1]

    perm = torch.randperm(num_edges)

    train_size = int(
        0.8 * num_edges
    )

    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    train_edges = data.edge_index[
        :, train_idx
    ]

    test_edges = data.edge_index[
        :, test_idx
    ]

    train_types = type_labels[
        train_idx.numpy()
    ]

    test_types = type_labels[
        test_idx.numpy()
    ]

    print("\nTrain type counts:")
    print(
        pd.Series(train_types)
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nTest type counts:")
    print(
        pd.Series(test_types)
        .value_counts(dropna=False)
        .sort_index()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MultiTaskGNN(
        input_dim=data.x.shape[1],
        hidden_dim=64,
        num_classes=3
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        # IMPORTANT:
        # Only training edges are used for message passing.
        z = model.encode(
            data.x,
            train_edges
        )

        # ====================================================
        # TASK 1: LINK PREDICTION
        # ====================================================

        positive_scores = model.predict_link(
            z,
            train_edges
        )

        negative_edges = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.shape[1]
        )

        negative_scores = model.predict_link(
            z,
            negative_edges
        )

        link_scores = torch.cat([
            positive_scores,
            negative_scores
        ])

        link_labels = torch.cat([
            torch.ones_like(
                positive_scores
            ),
            torch.zeros_like(
                negative_scores
            )
        ])

        link_loss = (
            torch.nn.functional
            .binary_cross_entropy_with_logits(
                link_scores,
                link_labels
            )
        )

        # ====================================================
        # TASK 2: INTERACTION TYPE
        # ====================================================

        labeled_mask = ~np.isnan(
            train_types
        )

        if labeled_mask.sum() > 0:

            labeled_positions = torch.tensor(
                np.where(labeled_mask)[0],
                dtype=torch.long,
                device=device
            )

            labeled_edges = train_edges[
                :,
                labeled_positions
            ]

            type_targets = torch.tensor(
                train_types[labeled_mask],
                dtype=torch.long,
                device=device
            )

            type_logits = model.predict_type(
                z,
                labeled_edges
            )

            type_loss = (
                torch.nn.functional
                .cross_entropy(
                    type_logits,
                    type_targets
                )
            )

        else:

            type_loss = torch.tensor(
                0.0,
                device=device
            )

        # ----------------------------------------------------
        # Joint loss
        # ----------------------------------------------------

        loss = (
            link_loss +
            type_loss
        )

        loss.backward()

        optimizer.step()

    # ========================================================
    # EVALUATION
    # ========================================================

    model.eval()

    with torch.no_grad():

        z = model.encode(
            data.x,
            train_edges
        )

        # ====================================================
        # LINK PREDICTION
        # ====================================================

        positive_scores = model.predict_link(
            z,
            test_edges
        )

        negative_edges = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=test_edges.shape[1]
        )

        negative_scores = model.predict_link(
            z,
            negative_edges
        )

        scores = torch.cat([
            positive_scores,
            negative_scores
        ]).cpu().numpy()

        labels = np.concatenate([
            np.ones(
                len(positive_scores)
            ),
            np.zeros(
                len(negative_scores)
            )
        ])

        predictions = (
            scores >= 0
        ).astype(int)

        link_auc = roc_auc_score(
            labels,
            scores
        )

        link_ap = average_precision_score(
            labels,
            scores
        )

        link_precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )

        link_recall = recall_score(
            labels,
            predictions,
            zero_division=0
        )

        link_f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        # ====================================================
        # INTERACTION TYPE
        # ====================================================

        type_mask = ~np.isnan(
            test_types
        )

        labeled_test_types = (
            test_types[type_mask]
            .astype(int)
        )

        # ----------------------------------------------------
        # Only calculate multi-class metrics when at least
        # two classes are actually present in the test set.
        # ----------------------------------------------------

        unique_test_classes = np.unique(
            labeled_test_types
        )

        if (
            type_mask.sum() > 0
            and len(unique_test_classes) >= 2
        ):

            labeled_positions = torch.tensor(
                np.where(type_mask)[0],
                dtype=torch.long,
                device=device
            )

            labeled_test_edges = test_edges[
                :,
                labeled_positions
            ]

            type_logits = model.predict_type(
                z,
                labeled_test_edges
            )

            type_predictions = (
                torch.argmax(
                    type_logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

            type_accuracy = accuracy_score(
                labeled_test_types,
                type_predictions
            )

            type_precision = precision_score(
                labeled_test_types,
                type_predictions,
                average="macro",
                zero_division=0
            )

            type_recall = recall_score(
                labeled_test_types,
                type_predictions,
                average="macro",
                zero_division=0
            )

            type_f1 = f1_score(
                labeled_test_types,
                type_predictions,
                average="macro",
                zero_division=0
            )

        else:

            type_accuracy = np.nan
            type_precision = np.nan
            type_recall = np.nan
            type_f1 = np.nan

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nLINK PREDICTION")

    print(
        f"ROC-AUC: {link_auc:.4f}"
    )

    print(
        f"Average Precision: {link_ap:.4f}"
    )

    print(
        f"Precision: {link_precision:.4f}"
    )

    print(
        f"Recall: {link_recall:.4f}"
    )

    print(
        f"F1: {link_f1:.4f}"
    )

    print("\nINTERACTION TYPE")

    if np.isnan(type_f1):

        print(
            "Not evaluated: "
            "test set contains only one labeled class."
        )

    else:

        print(
            f"Accuracy: {type_accuracy:.4f}"
        )

        print(
            f"Macro Precision: {type_precision:.4f}"
        )

        print(
            f"Macro Recall: {type_recall:.4f}"
        )

        print(
            f"Macro F1: {type_f1:.4f}"
        )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    RESULTS.append({
        "seed": seed,
        "network_id": network_id,

        "link_roc_auc": link_auc,
        "link_average_precision": link_ap,
        "link_precision": link_precision,
        "link_recall": link_recall,
        "link_f1": link_f1,

        "type_accuracy": type_accuracy,
        "type_macro_precision": type_precision,
        "type_macro_recall": type_recall,
        "type_macro_f1": type_f1
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    networks = [
        2472,
        2474,
        5027,
        5028,
        5029
    ]

    seeds = [
        42,
        43,
        44,
        45,
        46
    ]

    # --------------------------------------------------------
    # Run all experiments
    # --------------------------------------------------------

    for seed in seeds:

        print("\n" + "=" * 60)
        print(f"SEED: {seed}")
        print("=" * 60)

        for network_id in networks:

            train_and_evaluate(
                network_id,
                seed
            )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        RESULTS
    )

    raw_file = (
        RESULTS_DIR /
        "multitask_results_5seeds.csv"
    )

    results_df.to_csv(
        raw_file,
        index=False
    )

    # ========================================================
    # CALCULATE MEAN + STANDARD DEVIATION
    # ========================================================

    metric_columns = [
        "link_roc_auc",
        "link_average_precision",
        "link_precision",
        "link_recall",
        "link_f1",
        "type_accuracy",
        "type_macro_precision",
        "type_macro_recall",
        "type_macro_f1"
    ]

    summary_rows = []

    for network_id, group in (
        results_df
        .groupby("network_id")
    ):

        row = {
            "network_id": network_id,
            "runs": len(group)
        }

        for metric in metric_columns:

            values = group[metric].dropna()

            if len(values) == 0:

                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan

            else:

                row[f"{metric}_mean"] = (
                    values.mean()
                )

                row[f"{metric}_std"] = (
                    values.std(ddof=1)
                    if len(values) > 1
                    else 0.0
                )

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        RESULTS_DIR /
        "multitask_results_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("MULTI-TASK EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Raw results rows: "
        f"{len(results_df)}"
    )

    print(
        f"Raw results: "
        f"{raw_file}"
    )

    print(
        f"Summary results: "
        f"{summary_file}"
    )

    print("\nSUMMARY")

    print(
        summary_df.to_string(
            index=False
        )
    )