from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data/results"
OUTPUT = RESULTS / "final_model_comparison.csv"

rows = [
    ["GCN", 2472, 0.6143, 0.5724, 0.5714, 0.7004, 0.6294],
    ["GCN", 2474, 0.6442, 0.5974, 0.6217, 0.6241, 0.6229],
    ["GCN", 5027, 0.7882, 0.7327, 0.6657, 0.8684, 0.7536],
    ["GCN", 5028, 0.7776, 0.7132, 0.6505, 0.8794, 0.7478],
    ["GCN", 5029, 0.7807, 0.7260, 0.6426, 0.8966, 0.7486],

    ["GAT", 2472, 0.6887, 0.6149, 0.6324, 0.7899, 0.7024],
    ["GAT", 2474, 0.6523, 0.6054, 0.6091, 0.8083, 0.6947],
    ["GAT", 5027, 0.7983, 0.7315, 0.6522, 0.9097, 0.7597],
    ["GAT", 5028, 0.7574, 0.7044, 0.6320, 0.8575, 0.7276],
    ["GAT", 5029, 0.7846, 0.7225, 0.6499, 0.8886, 0.7507],

    ["Graph Transformer", 2472, 0.6897, 0.6971, 0.6082, 0.7549, 0.6736],
    ["Graph Transformer", 2474, 0.6846, 0.6251, 0.6083, 0.8233, 0.6997],
    ["Graph Transformer", 5027, 0.8106, 0.7717, 0.6696, 0.8890, 0.7639],
    ["Graph Transformer", 5028, 0.7819, 0.7574, 0.6420, 0.8977, 0.7486],
    ["Graph Transformer", 5029, 0.8235, 0.7967, 0.6673, 0.8912, 0.7632],

    ["Ensemble", 2472, 0.6940, 0.6648, 0.6189, 0.7899, 0.6940],
    ["Ensemble", 2474, 0.7190, 0.6551, 0.6467, 0.8120, 0.7200],
    ["Ensemble", 5027, 0.8125, 0.7632, 0.6538, 0.8968, 0.7563],
    ["Ensemble", 5028, 0.8345, 0.8004, 0.6777, 0.8965, 0.7719],
    ["Ensemble", 5029, 0.8195, 0.7667, 0.6703, 0.8926, 0.7656],
]

columns = [
    "model",
    "network_id",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv(OUTPUT, index=False)

print("Final model comparison created.")
print(f"Rows: {len(df)}")
print(f"Saved: {OUTPUT}")
print("\nMean metrics by model:")
print(
    df.groupby("model")[
        ["roc_auc", "average_precision", "precision", "recall", "f1"]
    ].mean().round(4).to_string()
)