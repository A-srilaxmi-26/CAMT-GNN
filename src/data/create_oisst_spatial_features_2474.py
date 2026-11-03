import xarray as xr
import numpy as np
import pandas as pd

INPUT = "data/raw/oisst/oisst_2007_06_01_network_2474.nc"
OUTPUT = "data/processed/network_2474_climate_spatial_features.csv"

ds = xr.open_dataset(INPUT)

sst = ds["sst"].isel(time=0, zlev=0)

values = sst.values.astype(float)
lat = sst["lat"].values.astype(float)
lon = sst["lon"].values.astype(float)

# Remove missing values for statistical features
valid = values[np.isfinite(values)]

# Basic spatial SST statistics
sst_mean = np.mean(valid)
sst_min = np.min(valid)
sst_max = np.max(valid)
sst_std = np.std(valid)
sst_range = sst_max - sst_min

# Spatial gradients
# np.gradient returns change along latitude and longitude axes.
lat_gradient = np.gradient(values, lat, axis=0)
lon_gradient = np.gradient(values, lon, axis=1)

lat_gradient_valid = lat_gradient[np.isfinite(lat_gradient)]
lon_gradient_valid = lon_gradient[np.isfinite(lon_gradient)]

features = {
    "network_id": 2474,
    "time": str(ds["time"].values[0]),
    "sst_mean": sst_mean,
    "sst_min": sst_min,
    "sst_max": sst_max,
    "sst_std": sst_std,
    "sst_range": sst_range,
    "sst_lat_gradient_mean": np.mean(np.abs(lat_gradient_valid)),
    "sst_lon_gradient_mean": np.mean(np.abs(lon_gradient_valid)),
    "grid_cells": len(valid),
}

df = pd.DataFrame([features])

df.to_csv(OUTPUT, index=False)

print("=" * 70)
print("CAMT-GNN | OISST SPATIAL FEATURES | NETWORK 2472")
print("=" * 70)

print("\nGenerated features:")
print(df.to_string(index=False))

print("\nSaved to:")
print(OUTPUT)

print("\nSUCCESS")