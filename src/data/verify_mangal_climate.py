from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "mangal_climate_networks.csv"
)


print("=" * 70)
print("CAMT-GNN | VERIFY MANGAL + NOAA CLIMATE DATASET")
print("=" * 70)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    print("\nERROR: File not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


df = pd.read_csv(INPUT_FILE)


# ------------------------------------------------------------
# BASIC CHECKS
# ------------------------------------------------------------

print("\nRows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nNetwork IDs:")
print(df["network_id"].tolist())


# ------------------------------------------------------------
# DUPLICATES
# ------------------------------------------------------------

print(
    "\nDuplicate network IDs:",
    df["network_id"].duplicated().sum()
)


# ------------------------------------------------------------
# CLIMATE COVERAGE
# ------------------------------------------------------------

climate_columns = [
    "sst_mean",
    "sst_min",
    "sst_max",
    "sst_std",
    "grid_cells"
]

print("\nClimate missing values:")

print(
    df[climate_columns].isna().sum()
)


print("\nNetworks with climate data:")

print(
    df[
        df["sst_mean"].notna()
    ][
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
# CHECK CLIMATE CONSISTENCY
# ------------------------------------------------------------

valid = df["sst_mean"].notna()

if valid.any():

    print("\nClimate consistency checks:")

    print(
        "Minimum SST <= Mean SST:",
        (
            df.loc[valid, "sst_min"]
            <= df.loc[valid, "sst_mean"]
        ).all()
    )

    print(
        "Mean SST <= Maximum SST:",
        (
            df.loc[valid, "sst_mean"]
            <= df.loc[valid, "sst_max"]
        ).all()
    )

    print(
        "Grid cells > 0:",
        (
            df.loc[valid, "grid_cells"] > 0
        ).all()
    )


print("\n" + "=" * 70)
print("MANGAL + NOAA VERIFICATION COMPLETE")
print("=" * 70)