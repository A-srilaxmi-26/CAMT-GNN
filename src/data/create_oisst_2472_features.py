from pathlib import Path

import xarray as xr
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "oisst"
    / "oisst_2005_06_01_network_2472.nc"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "oisst_2472_2005_features.csv"
)


print("=" * 70)
print("CAMT-GNN | NETWORK 2472 | NOAA OISST FEATURES")
print("=" * 70)


if not INPUT_FILE.exists():
    print("\nERROR: Input file not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


ds = xr.open_dataset(
    INPUT_FILE,
    engine="netcdf4"
)

sst = ds["sst"]

df = sst.to_dataframe(
    name="sst"
).reset_index()

df = df.dropna(
    subset=["sst"]
).copy()

df = df[
    [
        "time",
        "lat",
        "lon",
        "sst"
    ]
]

df["network_id"] = 2472

df = df[
    [
        "network_id",
        "time",
        "lat",
        "lon",
        "sst"
    ]
]

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nRows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nSST statistics:")
print("Minimum :", df["sst"].min())
print("Maximum :", df["sst"].max())
print("Mean    :", df["sst"].mean())

print("\nNetwork IDs:")
print(df["network_id"].unique())

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved:")
print(OUTPUT_FILE)

ds.close()

print("\n" + "=" * 70)
print("SUCCESS — NETWORK 2472 OISST FEATURES CREATED")
print("=" * 70)