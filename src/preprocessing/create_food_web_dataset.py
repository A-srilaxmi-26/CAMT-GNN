from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "mangal_interactions.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "food_web_edges.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("CAMT-GNN | CREATE STANDARDIZED FOOD-WEB DATASET")
print("=" * 70)

print(f"\nInput interactions: {len(df)}")


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

food_web = df[
    [
        "network_id",
        "interaction_id",
        "node_from",
        "node_to",
        "date",
        "direction",
        "type",
        "method",
        "attr_id",
        "value",
    ]
].copy()


# ============================================================
# ADD EXPLICIT EDGE COLUMNS
# ============================================================

food_web["source_node"] = food_web["node_from"]

food_web["target_node"] = food_web["node_to"]

food_web["interaction_type"] = food_web["type"]


# ============================================================
# SAVE
# ============================================================

food_web.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print(
    f"\nOutput interactions: {len(food_web)}"
)

print(
    f"Output columns: {len(food_web.columns)}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\nSample:")

print(
    food_web.head(10).to_string(
        index=False
    )
)

print("\n✅ Food-web dataset created.")