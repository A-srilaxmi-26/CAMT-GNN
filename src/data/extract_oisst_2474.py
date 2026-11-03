import xarray as xr
import os

INPUT = "data/raw/oisst/oisst_20070601.nc"
OUTPUT = "data/raw/oisst/oisst_2007_06_01_network_2474.nc"

LAT_MIN = 41.875
LAT_MAX = 48.375

LON_MIN = 233.875
LON_MAX = 236.125

ds = xr.open_dataset(INPUT)

subset = ds.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)

os.makedirs(
    "data/raw/oisst",
    exist_ok=True
)

subset.to_netcdf(OUTPUT)

print("=" * 70)
print("CAMT-GNN | OISST SUBSET | NETWORK 2474")
print("=" * 70)

print("\nRequested region:")
print(f"Latitude : {LAT_MIN} to {LAT_MAX}")
print(f"Longitude: {LON_MIN} to {LON_MAX}")

print("\nActual subset:")
print(subset)

print("\nSST range:")

sst = subset["sst"]

print(
    "min:",
    float(sst.min())
)

print(
    "max:",
    float(sst.max())
)

print(
    "mean:",
    float(sst.mean())
)

print("\nSaved:")
print(OUTPUT)

print("\nSUCCESS")