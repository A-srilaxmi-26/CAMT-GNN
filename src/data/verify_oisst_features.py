from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "oisst_2005_06_01_features.csv"
)


print("=" * 70)
print("CAMT-GNN | VERIFY NOAA OISST FEATURES")
print("=" * 70)


if not INPUT_FILE.exists():
    print("\nERROR: File not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


df = pd.read_csv(INPUT_FILE)

print("\nRows:", len(df))
print("Columns:", df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nUnique dates:")
print(df["time"].unique())

print("\nLatitude:")
print(df["lat"].min(), "to", df["lat"].max())

print("\nLongitude:")
print(df["lon"].min(), "to", df["lon"].max())

print("\nSST:")
print(df["sst"].min(), "to", df["sst"].max())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nSample:")
print(df.head())

print("\n" + "=" * 70)
print("OISST VERIFICATION COMPLETE")
print("=" * 70)