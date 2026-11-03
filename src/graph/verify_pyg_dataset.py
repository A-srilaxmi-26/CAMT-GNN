from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/pyg"

for file in sorted(DATA_DIR.glob("*.pt")):
    data = torch.load(file, weights_only=False)

    print(
        f"{file.name}: "
        f"nodes={data.num_nodes}, "
        f"edges={data.num_edges}, "
        f"x_shape={tuple(data.x.shape)}, "
        f"edge_index_shape={tuple(data.edge_index.shape)}"
    )