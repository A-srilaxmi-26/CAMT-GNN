from pathlib import Path

import xarray as xr
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "oisst"
    / "oisst_2005_06_01_northern_california_subset.nc"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "oisst_2005_06_01_features.csv"
)


print("=" * 70)
print("CAMT-GNN | NOAA OISST FEATURE EXTRACTION")
print("=" * 70)


# ------------------------------------------------------------
# CHECK FILE
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    print("\nERROR: OISST subset not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


print("\nInput:")
print(INPUT_FILE)


# ------------------------------------------------------------
# OPEN DATASET
# ------------------------------------------------------------

ds = xr.open_dataset(
    INPUT_FILE,
    engine="netcdf4"
)

sst = ds["sst"]


# ------------------------------------------------------------
# CONVERT TO DATAFRAME
# ------------------------------------------------------------

df = sst.to_dataframe(
    name="sst"
).reset_index()


# ------------------------------------------------------------
# REMOVE MISSING VALUES
# ------------------------------------------------------------

df = df.dropna(subset=["sst"]).copy()


# ------------------------------------------------------------
# KEEP RELEVANT COLUMNS
# ------------------------------------------------------------

df = df[
    [
        "time",
        "lat",
        "lon",
        "sst"
    ]
]


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\nRows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nSST statistics:")
print("Minimum :", df["sst"].min())
print("Maximum :", df["sst"].max())
print("Mean    :", df["sst"].mean())

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved:")
print(OUTPUT_FILE)

ds.close()


print("\n" + "=" * 70)
print("SUCCESS — NOAA OISST FEATURES CREATED")
print("=" * 70)