from pathlib import Path

import xarray as xr


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "oisst"

INPUT_FILE = RAW_DIR / "oisst_20050601.nc"


# ============================================================
# CAMT-GNN REGION
# Northern California Current
#
# IMPORTANT:
# NOAA OISST longitude is 0-360 degrees.
#
# -128 -> 232
# -124 -> 236
# ============================================================

LAT_MIN = 40.0
LAT_MAX = 43.0

LON_MIN = 232.0
LON_MAX = 236.0


print("=" * 70)
print("CAMT-GNN | NOAA OISST v2.1 | REAL DATA")
print("=" * 70)


# ============================================================
# CHECK FILE
# ============================================================

if not INPUT_FILE.exists():
    print("\nERROR: NOAA file does not exist:")
    print(INPUT_FILE)
    raise SystemExit(1)


print("\nUsing downloaded NOAA file:")
print(INPUT_FILE)

print(
    "\nFile size:",
    round(INPUT_FILE.stat().st_size / (1024 * 1024), 2),
    "MB"
)


# ============================================================
# OPEN DATASET
# ============================================================

print("\nOpening NetCDF...")

ds = xr.open_dataset(
    INPUT_FILE,
    engine="netcdf4"
)

print("\nDimensions:")
print(ds.dims)

print("\nVariables:")
print(list(ds.data_vars))


# ============================================================
# SHOW COORDINATE RANGE
# ============================================================

print("\nLatitude range:")
print(
    float(ds.lat.min()),
    "to",
    float(ds.lat.max())
)

print("\nLongitude range:")
print(
    float(ds.lon.min()),
    "to",
    float(ds.lon.max())
)


# ============================================================
# SELECT REAL REGION
# ============================================================

print("\nSelecting Northern California region...")

sst = ds["sst"].sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)


# ============================================================
# LOAD SUBSET
# ============================================================

sst = sst.load()

print("\nSubset shape:")
print(sst.shape)


# ============================================================
# VALIDATION
# ============================================================

if sst.size == 0:
    print("\nERROR: Empty SST subset.")
    raise SystemExit(1)


print("\nSST metadata:")
print(sst.attrs)


# ============================================================
# STATISTICS
# ============================================================

print("\nREAL NOAA SST statistics:")

print("Minimum :", float(sst.min()))
print("Maximum :", float(sst.max()))
print("Mean    :", float(sst.mean()))


# ============================================================
# SAVE SUBSET
# ============================================================

OUTPUT_FILE = (
    RAW_DIR /
    "oisst_2005_06_01_northern_california_subset.nc"
)

sst.to_netcdf(OUTPUT_FILE)

ds.close()


# ============================================================
# FINAL
# ============================================================

print("\nSubset saved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("SUCCESS — REAL NOAA OISST DATA READY")
print("=" * 70)