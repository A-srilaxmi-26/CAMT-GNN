from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mangal_nodes.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "food_web_nodes.csv"

df = pd.read_csv(INPUT_FILE)

nodes = df[
    [
        "network_id",
        "node_id",
        "original_name",
        "node_level",
        "taxonomy_id",
        "taxonomy_name",
    ]
].copy()

nodes["species_name"] = nodes["original_name"]

nodes.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 70)
print("CAMT-GNN | STANDARDIZED NODE DATASET")
print("=" * 70)

print(f"\nInput nodes: {len(df)}")
print(f"Output nodes: {len(nodes)}")

print(f"\nSaved to:\n{OUTPUT_FILE}")

print("\n✅ Node dataset created.")