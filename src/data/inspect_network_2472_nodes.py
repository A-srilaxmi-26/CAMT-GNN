import pandas as pd

path = "data/processed/food_web_nodes.csv"

df = pd.read_csv(path)

# Filter network 2472
net = df[df["network_id"] == 2472].copy()

print("=" * 70)
print("CAMT-GNN | NODE METADATA INSPECTION | NETWORK 2472")
print("=" * 70)

print("\nShape:")
print(net.shape)

print("\nColumns:")
for col in net.columns:
    print(" -", col)

print("\n\nData types:")
print(net.dtypes)

print("\n\nMissing values:")
print(net.isna().sum())

print("\n\nFirst 10 rows:")
print(net.head(10).to_string())

print("\n\nUnique values / cardinality:")
for col in net.columns:
    print(f"{col}: {net[col].nunique(dropna=True)} unique")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)