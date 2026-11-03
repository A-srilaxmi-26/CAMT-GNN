from pathlib import Path

import xarray as xr


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "oisst"
    / "oisst_20050601.nc"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "oisst"
    / "oisst_2005_06_01_network_2472.nc"
)


# Mangal network 2472 polygon:
# latitude 42.00 to 48.34
# longitude -126.00 to -124.00
#
# Expanded slightly to include surrounding OISST grid cells.

LAT_MIN = 41.875
LAT_MAX = 48.375

LON_MIN = 233.875   # -126.125
LON_MAX = 236.125   # -123.875


print("=" * 70)
print("CAMT-GNN | NOAA OISST | NETWORK 2472 REGION")
print("=" * 70)


if not INPUT_FILE.exists():
    print("\nERROR: NOAA file not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


print("\nOpening:")
print(INPUT_FILE)

ds = xr.open_dataset(
    INPUT_FILE,
    engine="netcdf4"
)


print("\nSelecting network 2472 region...")

sst = ds["sst"].sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
).load()


print("\nSubset shape:")
print(sst.shape)


if sst.size == 0:
    print("\nERROR: Empty subset.")
    raise SystemExit(1)


print("\nActual selected latitude:")
print(
    float(sst.lat.min()),
    "to",
    float(sst.lat.max())
)

print("\nActual selected longitude:")
print(
    float(sst.lon.min()),
    "to",
    float(sst.lon.max())
)


print("\nSST statistics:")
print("Minimum :", float(sst.min()))
print("Maximum :", float(sst.max()))
print("Mean    :", float(sst.mean()))


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

sst.to_netcdf(OUTPUT_FILE)

ds.close()


print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("SUCCESS — NETWORK 2472 NOAA REGION CREATED")
print("=" * 70)