from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

edges_file = ROOT / "data" / "processed" / "food_web_edges.csv"
nodes_file = ROOT / "data" / "processed" / "food_web_nodes.csv"
networks_file = ROOT / "data" / "processed" / "food_web_networks.csv"


edges = pd.read_csv(edges_file)
nodes = pd.read_csv(nodes_file)
networks = pd.read_csv(networks_file)


print("\n===== MULTI-TASK TARGET CHECK =====")

print("\n1. Food-web target")
print(f"Interactions: {len(edges)}")
print(f"Interaction types:")
print(edges["interaction_type"].value_counts(dropna=False))


print("\n2. Temporal target")

edges["date"] = pd.to_datetime(edges["date"], errors="coerce")

print(f"Total dated interactions: {edges['date'].notna().sum()}")
print(f"Missing dates: {edges['date'].isna().sum()}")

print("\nDates by network:")
print(
    edges.groupby("network_id")["date"]
    .agg(
        dated_interactions=lambda x: x.notna().sum(),
        first_date="min",
        last_date="max"
    )
)


print("\n3. Network information")

print("Available network columns:")
print(list(networks.columns))

print("\nNetwork records:")
print(networks.to_string(index=False))


print("\n4. Node information")

print(f"Nodes: {len(nodes)}")
print("Available node columns:")
print(list(nodes.columns))


print("\n===== CHECK COMPLETE =====")