from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

edges = pd.read_csv(
    ROOT / "data/processed/food_web_edges.csv"
)

temporal = edges.dropna(subset=["date"]).copy()

temporal["date"] = pd.to_datetime(temporal["date"], utc=True)

temporal = temporal.sort_values(
    ["network_id", "date", "interaction_id"]
)

output = ROOT / "data/processed/temporal_food_web_edges.csv"

temporal.to_csv(output, index=False)

print("Temporal edges:", len(temporal))
print("Networks:", temporal["network_id"].unique().tolist())
print("Dates:", temporal["date"].dt.year.unique().tolist())
print("Saved:", output)