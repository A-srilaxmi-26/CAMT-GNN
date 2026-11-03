from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "oisst_2472_2005_features.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "network_2472_climate_features.csv"
)


print("=" * 70)
print("CAMT-GNN | NETWORK 2472 | CLIMATE AGGREGATION")
print("=" * 70)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    print("\nERROR: Input file not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


df = pd.read_csv(INPUT_FILE)

print("\nInput rows:", len(df))


# ------------------------------------------------------------
# AGGREGATE REAL SST
# ------------------------------------------------------------

climate = (
    df.groupby(
        ["network_id", "time"],
        as_index=False
    )
    .agg(
        sst_mean=("sst", "mean"),
        sst_min=("sst", "min"),
        sst_max=("sst", "max"),
        sst_std=("sst", "std"),
        grid_cells=("sst", "count")
    )
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

climate.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\nNetwork-level climate features:")
print(climate.to_string(index=False))

print("\nSaved:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("SUCCESS — NETWORK CLIMATE FEATURES CREATED")
print("=" * 70)