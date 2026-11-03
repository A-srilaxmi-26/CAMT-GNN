from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mangal_networks.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "food_web_networks.csv"

df = pd.read_csv(INPUT_FILE)

networks = df[
    [
        "network_id",
        "network_name",
        "network_date",
        "description",
        "latitude",
        "longitude",
    ]
].copy()

networks.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 70)
print("CAMT-GNN | NETWORK METADATA DATASET")
print("=" * 70)

print(f"\nNetworks: {len(networks)}")

print(f"\nSaved to:\n{OUTPUT_FILE}")

print("\n✅ Network dataset created.")