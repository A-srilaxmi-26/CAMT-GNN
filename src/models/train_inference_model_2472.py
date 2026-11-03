import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling


# ============================================================
# CAMT-GNN | INFERENCE MODEL TRAINING
# NETWORK 2472
#
# Purpose:
# Train a real multi-task GNN and save its checkpoint
# for later Streamlit inference.
#
# Tasks:
#   1. Link prediction
#   2. Interaction-type classification
#
# Real data:
#   Mangal food-web network 2472
#
# IMPORTANT:
# This script does NOT fabricate predictions or metrics.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pyg"
    / "network_2472.pt"
)

NODES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "food_web_nodes.csv"
)

EDGES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "multitask_edges.csv"
)

CLIMATE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "network_2472_climate_spatial_features.csv"
)

CHECKPOINT_FILE = (
    ROOT
    / "data"
    / "results"
    / "camt_gnn_2472_inference.pt"
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 200

LEARNING_RATE = 0.01

HIDDEN_DIM = 64

NUM_CLASSES = 3

CLIMATE_DIM = 7

CLIMATE_WEIGHT = 0.25

# ============================================================
# INTERACTION TYPE MAPPING
# ============================================================

TYPE_NAMES = {
    0: "predation",
    1: "herbivory",
    2: "detritivore",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# MODEL
# ============================================================

class CAMTGNNInference(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_classes=3,
        climate_dim=7,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Food-web graph encoder
        # ----------------------------------------------------

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        # ----------------------------------------------------
        # Climate encoder
        # ----------------------------------------------------

        self.climate_encoder = nn.Sequential(

            nn.Linear(
                climate_dim,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU()
        )

        # ----------------------------------------------------
        # Climate + graph fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                hidden_dim + 16,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )

        # ----------------------------------------------------
        # Link prediction head
        # ----------------------------------------------------

        self.link_head = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

        # ----------------------------------------------------
        # Interaction-type head
        # ----------------------------------------------------

        self.type_head = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    # ========================================================
    # ENCODER
    # ========================================================

    def encode(
        self,
        x,
        edge_index,
        climate
    ):

        h = self.conv1(
            x,
            edge_index
        )

        h = F.relu(h)

        h = self.conv2(
            h,
            edge_index
        )

        h = F.relu(h)

        # ----------------------------------------------------
        # Climate is graph-level context.
        # ----------------------------------------------------

        climate_embedding = self.climate_encoder(
            climate
        )

        climate_embedding = climate_embedding.unsqueeze(0)

        climate_embedding = climate_embedding.expand(
            h.shape[0],
            -1
        )

        # ----------------------------------------------------
        # Fuse
        # ----------------------------------------------------

        h = torch.cat(
            [
                h,
                climate_embedding
            ],
            dim=1
        )

        h = self.fusion(h)

        return h

    # ========================================================
    # EDGE REPRESENTATION
    # ========================================================

    def edge_representation(
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

        return torch.cat(
            [
                source,
                target
            ],
            dim=1
        )

    # ========================================================
    # LINK PREDICTION
    # ========================================================

    def predict_link(
        self,
        z,
        edge_index
    ):

        edge_repr = self.edge_representation(
            z,
            edge_index
        )

        return self.link_head(
            edge_repr
        ).view(-1)

    # ========================================================
    # INTERACTION TYPE
    # ========================================================

    def predict_type(
        self,
        z,
        edge_index
    ):

        edge_repr = self.edge_representation(
            z,
            edge_index
        )

        return self.type_head(
            edge_repr
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)

    print(
        "CAMT-GNN | TRAIN INFERENCE MODEL | NETWORK 2472"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    data = torch.load(
        GRAPH_FILE,
        weights_only=False
    )

    data.x = data.x.float()

    data.edge_index = data.edge_index.long()

    print("\nGraph:")

    print(
        "Nodes:",
        data.num_nodes
    )

    print(
        "Edges:",
        data.edge_index.shape[1]
    )

    print(
        "Features:",
        data.x.shape
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    nodes_df = pd.read_csv(
        NODES_FILE
    )

    nodes_df = nodes_df[
        nodes_df["network_id"] == 2472
    ].copy()

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edges_df = pd.read_csv(
        EDGES_FILE
    )

    edges_df = edges_df[
        edges_df["network_id"] == 2472
    ].copy()

    # --------------------------------------------------------
    # Climate
    # --------------------------------------------------------

    climate_df = pd.read_csv(
        CLIMATE_FILE
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
        climate_df[
            climate_columns
        ]
        .iloc[0]
        .values
        .astype(np.float32)
    )

    print("\nClimate:")

    for name, value in zip(
        climate_columns,
        climate.numpy()
    ):

        print(
            f"{name}: {value:.6f}"
        )

    return (
        data,
        nodes_df,
        edges_df,
        climate
    )


# ============================================================
# BUILD EDGE TYPE LABELS
# ============================================================

def build_type_labels(
    data,
    nodes_df,
    edges_df
):

    # --------------------------------------------------------
    # Exact node mapping used by PyG dataset
    # --------------------------------------------------------

    node_ids = nodes_df[
        "node_id"
    ].tolist()

    node_to_idx = {

        node_id: index

        for index, node_id
        in enumerate(node_ids)
    }

    # --------------------------------------------------------
    # Edge → interaction type
    # --------------------------------------------------------

    edge_type_map = {}

    for _, row in edges_df.iterrows():

        source = row["node_from"]

        target = row["node_to"]

        if (
            source not in node_to_idx
            or target not in node_to_idx
        ):
            continue

        key = (
            node_to_idx[source],
            node_to_idx[target]
        )

        value = row[
            "interaction_type_target"
        ]

        edge_type_map[key] = value

    # --------------------------------------------------------
    # Match labels to graph edge order
    # --------------------------------------------------------

    labels = []

    graph_edges = (
        data.edge_index
        .t()
        .cpu()
        .numpy()
    )

    for source, target in graph_edges:

        key = (
            int(source),
            int(target)
        )

        labels.append(
            edge_type_map.get(
                key,
                np.nan
            )
        )

    labels = np.array(
        labels,
        dtype=float
    )

    return labels


# ============================================================
# TRAINING
# ============================================================

def train_model():

    set_seed(SEED)

    (
        data,
        nodes_df,
        edges_df,
        climate
    ) = load_data()

    type_labels = build_type_labels(
        data,
        nodes_df,
        edges_df
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    num_edges = (
        data.edge_index.shape[1]
    )

    permutation = torch.randperm(
        num_edges
    )

    train_size = int(
        0.8 * num_edges
    )

    train_idx = permutation[
        :train_size
    ]

    test_idx = permutation[
        train_size:
    ]

    train_edges = data.edge_index[
        :,
        train_idx
    ]

    test_edges = data.edge_index[
        :,
        test_idx
    ]

    train_types = type_labels[
        train_idx.numpy()
    ]

    test_types = type_labels[
        test_idx.numpy()
    ]

    print("\nSplit:")

    print(
        "Train edges:",
        train_edges.shape[1]
    )

    print(
        "Test edges:",
        test_edges.shape[1]
    )

    print(
        "Train labeled types:",
        np.sum(
            ~np.isnan(train_types)
        )
    )

    print(
        "Test labeled types:",
        np.sum(
            ~np.isnan(test_types)
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CAMTGNNInference(
        input_dim=data.x.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        climate_dim=CLIMATE_DIM
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining...")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Encode using training graph only
        # ----------------------------------------------------

        z = model.encode(
            data.x,
            train_edges,
            climate
        )

        # ====================================================
        # TASK 1 — LINK PREDICTION
        # ====================================================

        positive_logits = model.predict_link(
            z,
            train_edges
        )

        negative_edges = negative_sampling(
            edge_index=train_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.shape[1]
        )

        negative_logits = model.predict_link(
            z,
            negative_edges
        )

        link_logits = torch.cat(
            [
                positive_logits,
                negative_logits
            ]
        )

        link_labels = torch.cat(
            [
                torch.ones_like(
                    positive_logits
                ),
                torch.zeros_like(
                    negative_logits
                )
            ]
        )

        link_loss = F.binary_cross_entropy_with_logits(
            link_logits,
            link_labels
        )

        # ====================================================
        # TASK 2 — INTERACTION TYPE
        # ====================================================

        labeled_mask = (
            ~np.isnan(train_types)
        )

        if labeled_mask.sum() > 0:

            positions = torch.tensor(
                np.where(labeled_mask)[0],
                dtype=torch.long
            )

            labeled_edges = train_edges[
                :,
                positions
            ]

            type_targets = torch.tensor(
                train_types[
                    labeled_mask
                ],
                dtype=torch.long
            )

            type_logits = model.predict_type(
                z,
                labeled_edges
            )

            type_loss = F.cross_entropy(
                type_logits,
                type_targets
            )

        else:

            type_loss = torch.tensor(
                0.0
            )

        # ----------------------------------------------------
        # Joint objective
        # ----------------------------------------------------

        loss = (
            link_loss
            + CLIMATE_WEIGHT * type_loss
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Total={loss.item():.5f} | "
                f"Link={link_loss.item():.5f} | "
                f"Type={type_loss.item():.5f}"
            )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "input_dim":
            data.x.shape[1],

        "hidden_dim":
            HIDDEN_DIM,

        "num_classes":
            NUM_CLASSES,

        "climate_dim":
            CLIMATE_DIM,

        "network_id":
            2472,

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "climate_weight":
            CLIMATE_WEIGHT,

        "climate_columns":
            [
                "sst_mean",
                "sst_min",
                "sst_max",
                "sst_std",
                "sst_range",
                "sst_lat_gradient_mean",
                "sst_lon_gradient_mean",
            ],

        "type_names":
            TYPE_NAMES,

        "node_count":
            data.num_nodes,

        "edge_count":
            data.edge_index.shape[1],
    }

    os.makedirs(
        CHECKPOINT_FILE.parent,
        exist_ok=True
    )

    torch.save(
        checkpoint,
        CHECKPOINT_FILE
    )

    print("\n" + "=" * 70)

    print(
        "MODEL TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        "\nCheckpoint saved:"
    )

    print(
        CHECKPOINT_FILE
    )

    print("\nSUCCESS")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_model()