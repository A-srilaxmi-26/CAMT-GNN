from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MANGAL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "food_web_networks.csv"
)

CLIMATE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "network_2472_climate_features.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "mangal_climate_networks.csv"
)


print("=" * 70)
print("CAMT-GNN | MANGAL + NOAA CLIMATE INTEGRATION")
print("=" * 70)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

mangal = pd.read_csv(MANGAL_FILE)
climate = pd.read_csv(CLIMATE_FILE)

print("\nMangal networks:", len(mangal))
print("Climate records:", len(climate))


# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

merged = mangal.merge(
    climate,
    on="network_id",
    how="left"
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\nMerged rows:", len(merged))

print("\nClimate coverage:")
print(
    merged[
        [
            "network_id",
            "network_name",
            "network_date",
            "sst_mean",
            "sst_min",
            "sst_max",
            "sst_std",
            "grid_cells"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("SUCCESS — MANGAL + NOAA DATASET CREATED")
print("=" * 70)