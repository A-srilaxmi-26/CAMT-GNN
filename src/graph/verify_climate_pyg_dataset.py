from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "pyg_climate"
    / "network_2472_climate.pt"
)


print("=" * 70)
print("CAMT-GNN | VERIFY CLIMATE-AWARE PyG DATASET")
print("=" * 70)


if not INPUT_FILE.exists():
    print("\nERROR: Climate PyG file not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


data = torch.load(
    INPUT_FILE,
    weights_only=False
)


print("\nNetwork ID:")
print(data.network_id)

print("\nNodes:")
print(data.num_nodes)

print("\nEdges:")
print(data.num_edges)

print("\nNode feature shape:")
print(tuple(data.x.shape))

print("\nClimate features:")

print("SST mean:", data.sst_mean)
print("SST min :", data.sst_min)
print("SST max :", data.sst_max)
print("SST std :", data.sst_std)


print("\nEdge index shape:")
print(tuple(data.edge_index.shape))


print("\nExpected checks:")

print(
    "78 nodes:",
    data.num_nodes == 78
)

print(
    "1283 edges:",
    data.num_edges == 1283
)

print(
    "82 features:",
    data.num_node_features == 82
)

print(
    "Climate values present:",
    all(
        value is not None
        for value in [
            data.sst_mean,
            data.sst_min,
            data.sst_max,
            data.sst_std
        ]
    )
)


print("\n" + "=" * 70)
print("CLIMATE PyG VERIFICATION COMPLETE")
print("=" * 70)