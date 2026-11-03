from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

edges = pd.read_csv(
    ROOT / "data/processed/food_web_edges.csv"
)

print("Total edges:", len(edges))
print("\nDate values:")
print(edges["date"].value_counts(dropna=False))

print("\nDate type:")
print(edges["date"].dtype)

print("\nNetworks with dates:")
print(
    edges.groupby("network_id")["date"]
    .apply(lambda x: x.notna().sum())
)