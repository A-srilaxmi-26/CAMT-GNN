import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ============================================================
# CONFIG
# ============================================================

NETWORK_ID = 2472

CHECKPOINT = "data/results/camt_gnn_2472_inference.pt"
GRAPH_PATH = "data/processed/pyg/network_2472.pt"
NODES_CSV = "data/processed/food_web_nodes.csv"
CLIMATE_CSV = "data/processed/network_2472_climate_spatial_features.csv"

TYPE_NAMES = {
    0: "predation",
    1: "herbivory",
    2: "detritivore"
}

CLIMATE_COLUMNS = [
    "sst_mean",
    "sst_min",
    "sst_max",
    "sst_std",
    "sst_range",
    "sst_lat_gradient_mean",
    "sst_lon_gradient_mean"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# MODEL
# ============================================================

class CAMTGNNInference(nn.Module):

    def __init__(
        self,
        input_dim,
        climate_dim=7,
        hidden_dim=64,
        climate_hidden=32,
        climate_out=16,
        num_types=3
    ):
        super().__init__()

        # Graph encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Climate encoder
        self.climate_encoder = nn.Sequential(
            nn.Linear(climate_dim, climate_hidden),
            nn.ReLU(),
            nn.Linear(climate_hidden, climate_out),
            nn.ReLU()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + climate_out, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        # Link prediction
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Interaction type classification
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types)
        )

    def encode(self, x, edge_index, climate):

        h = self.conv1(x, edge_index)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = F.relu(h)

        climate_embedding = self.climate_encoder(climate)

        # Same network-level climate context for every node
        climate_embedding = climate_embedding.expand(
            h.size(0),
            -1
        )

        h = torch.cat(
            [h, climate_embedding],
            dim=1
        )

        h = self.fusion(h)

        return h

    def predict_pair(self, h, source_idx, target_idx):

        source = h[source_idx]
        target = h[target_idx]

        pair = torch.cat(
            [source, target],
            dim=1
        )

        link_logit = self.link_head(pair).squeeze(-1)

        type_logits = self.type_head(pair)

        return link_logit, type_logits


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CAMT-GNN | REAL INFERENCE | NETWORK 2472")
print("=" * 70)


device = torch.device("cpu")

print("\nDevice:", device)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

print("\nCheckpoint loaded:")
print(CHECKPOINT)


# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------

if not os.path.exists(GRAPH_PATH):
    raise FileNotFoundError(
        f"Graph not found:\n{GRAPH_PATH}"
    )

graph = torch.load(
    GRAPH_PATH,
    map_location=device,
    weights_only=False
)

x = graph.x.float()
edge_index = graph.edge_index.long()

print("\nGraph:")
print("Nodes:", x.shape[0])
print("Edges:", edge_index.shape[1])
print("Features:", x.shape)


# ------------------------------------------------------------
# Load node metadata
# ------------------------------------------------------------

nodes_df = pd.read_csv(NODES_CSV)

nodes_df = nodes_df[
    nodes_df["network_id"] == NETWORK_ID
].copy()

nodes_df = nodes_df.reset_index(drop=True)

print("\nNode metadata:")
print("Nodes:", len(nodes_df))


# ------------------------------------------------------------
# Load climate
# ------------------------------------------------------------

climate_df = pd.read_csv(CLIMATE_CSV)

climate_row = climate_df[
    climate_df["network_id"] == NETWORK_ID
]

if climate_row.empty:
    raise ValueError(
        f"No climate record found for network {NETWORK_ID}"
    )

climate_values = (
    climate_row.iloc[0][CLIMATE_COLUMNS]
    .astype(float)
    .values
)

climate = torch.tensor(
    climate_values,
    dtype=torch.float32
).unsqueeze(0)

print("\nClimate:")
for column, value in zip(
    CLIMATE_COLUMNS,
    climate_values
):
    print(f"{column}: {value:.6f}")


# ============================================================
# CREATE MODEL
# ============================================================

model = CAMTGNNInference(
    input_dim=x.shape[1],
    climate_dim=len(CLIMATE_COLUMNS),
    hidden_dim=64,
    climate_hidden=32,
    climate_out=16,
    num_types=3
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("\nModel loaded successfully.")


# ============================================================
# GENERATE NODE EMBEDDINGS
# ============================================================

with torch.no_grad():

    h = model.encode(
        x,
        edge_index,
        climate
    )

print("\nNode embeddings:")
print(h.shape)


# ============================================================
# NODE LOOKUP
# ============================================================

def find_node(name):

    name = str(name).strip().lower()

    matches = nodes_df[
        nodes_df["original_name"]
        .astype(str)
        .str.lower()
        == name
    ]

    if matches.empty:

        matches = nodes_df[
            nodes_df["species_name"]
            .astype(str)
            .str.lower()
            == name
        ]

    if matches.empty:
        return None

    return int(matches.index[0])


# ============================================================
# INTERACTIVE PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("INTERACTION PREDICTION")
print("=" * 70)

print(
    "\nEnter an existing species/node from Network 2472."
)

print(
    "Type 'list' to see available species."
)

print(
    "Type 'exit' to stop."
)


while True:

    print("\n" + "-" * 70)

    predator_name = input(
        "Source / Predator species: "
    ).strip()

    if predator_name.lower() == "exit":
        break

    if predator_name.lower() == "list":

        print("\nAvailable nodes:\n")

        for i, row in nodes_df.iterrows():

            print(
                f"{i:3d} | "
                f"{row['original_name']}"
            )

        continue

    prey_name = input(
        "Target / Prey species: "
    ).strip()

    if prey_name.lower() == "exit":
        break

    source_idx = find_node(predator_name)
    target_idx = find_node(prey_name)

    if source_idx is None:

        print(
            "\nERROR: Source species was not found."
        )

        continue

    if target_idx is None:

        print(
            "\nERROR: Target species was not found."
        )

        continue


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    source_tensor = torch.tensor(
        [source_idx],
        dtype=torch.long
    )

    target_tensor = torch.tensor(
        [target_idx],
        dtype=torch.long
    )

    with torch.no_grad():

        link_logit, type_logits = model.predict_pair(
            h,
            source_tensor,
            target_tensor
        )

        link_probability = torch.sigmoid(
            link_logit
        ).item()

        type_probability = torch.softmax(
            type_logits,
            dim=1
        )[0]

        predicted_type_idx = int(
            torch.argmax(type_probability).item()
        )

        predicted_type = TYPE_NAMES[
            predicted_type_idx
        ]

        type_confidence = (
            type_probability[
                predicted_type_idx
            ].item()
        )


    # --------------------------------------------------------
    # Existing edge check
    # --------------------------------------------------------

    existing_edge = False

    src_edges = edge_index[0]
    dst_edges = edge_index[1]

    matches = (
        (src_edges == source_idx)
        &
        (dst_edges == target_idx)
    )

    if matches.any():
        existing_edge = True


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CAMT-GNN PREDICTION")
    print("=" * 70)

    print(
        f"\nSource / Predator : "
        f"{nodes_df.iloc[source_idx]['original_name']}"
    )

    print(
        f"Target / Prey     : "
        f"{nodes_df.iloc[target_idx]['original_name']}"
    )

    print(
        f"\nInteraction probability: "
        f"{link_probability:.4f}"
    )

    print(
        f"Interaction probability: "
        f"{link_probability * 100:.2f}%"
    )

    print(
        f"\nPredicted interaction type: "
        f"{predicted_type}"
    )

    print(
        f"Type confidence: "
        f"{type_confidence * 100:.2f}%"
    )

    print(
        f"\nKnown edge in dataset: "
        f"{'YES' if existing_edge else 'NO'}"
    )

    print("\nType probabilities:")

    for idx, name in TYPE_NAMES.items():

        probability = (
            type_probability[idx].item()
        )

        print(
            f"  {name:12s}: "
            f"{probability * 100:.2f}%"
        )

    print("\nClimate context used:")

    print(
        f"  SST mean  : "
        f"{climate_values[0]:.2f} °C"
    )

    print(
        f"  SST min   : "
        f"{climate_values[1]:.2f} °C"
    )

    print(
        f"  SST max   : "
        f"{climate_values[2]:.2f} °C"
    )

    print(
        f"  SST range : "
        f"{climate_values[4]:.2f} °C"
    )

    print("=" * 70)


print("\nInference finished.")