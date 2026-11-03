from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

NETWORK_FILE = (
    ROOT
    / "data"
    / "processed"
    / "food_web_networks.csv"
)

OISST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "oisst_2005_06_01_features.csv"
)


print("=" * 70)
print("CAMT-GNN | MANGAL + NOAA SPATIAL MATCH CHECK")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

networks = pd.read_csv(NETWORK_FILE)
oisst = pd.read_csv(OISST_FILE)


# ------------------------------------------------------------
# SELECT 2005 NETWORK
# ------------------------------------------------------------

network = networks[
    networks["network_id"] == 2472
].copy()


if network.empty:
    print("\nERROR: Network 2472 not found.")
    raise SystemExit(1)


print("\nMangal network:")
print(network.to_string(index=False))


# ------------------------------------------------------------
# NOAA RANGE
# ------------------------------------------------------------

print("\nNOAA OISST spatial range:")

print(
    "Latitude :",
    oisst["lat"].min(),
    "to",
    oisst["lat"].max()
)

print(
    "Longitude:",
    oisst["lon"].min(),
    "to",
    oisst["lon"].max()
)


# ------------------------------------------------------------
# CONVERT NOAA LONGITUDE TO -180 TO 180
# ------------------------------------------------------------

oisst["lon_deg"] = oisst["lon"].where(
    oisst["lon"] <= 180,
    oisst["lon"] - 360
)


print("\nNOAA longitude converted to -180 to 180:")

print(
    oisst["lon_deg"].min(),
    "to",
    oisst["lon_deg"].max()
)


# ------------------------------------------------------------
# CHECK
# ------------------------------------------------------------

print("\nSpatial matching check:")

print(
    "NOAA latitude range:",
    oisst["lat"].min(),
    "to",
    oisst["lat"].max()
)

print(
    "NOAA longitude range:",
    oisst["lon_deg"].min(),
    "to",
    oisst["lon_deg"].max()
)


print("\nNOTE:")
print(
    "Network 2472 contains a polygon-like geographic field "
    "rather than a single latitude/longitude point."
)

print(
    "Therefore, this script only verifies the coordinate systems "
    "and ranges. It does NOT assign SST values to the network yet."
)


print("\n" + "=" * 70)
print("SPATIAL MATCH CHECK COMPLETE")
print("=" * 70)