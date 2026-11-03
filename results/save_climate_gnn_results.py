from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REAL RESULTS FROM THE FIVE-SEED RUN
# ============================================================

results = [
    [2472, "ClimateAwareGNN", 42, 0.8758, 0.8567, 0.8210, 0.7344, 0.7753],
    [2472, "ClimateAwareGNN", 43, 0.9011, 0.8886, 0.7927, 0.8516, 0.8211],
    [2472, "ClimateAwareGNN", 44, 0.8865, 0.8574, 0.8051, 0.8555, 0.8295],
    [2472, "ClimateAwareGNN", 45, 0.9044, 0.8602, 0.8114, 0.8906, 0.8492],
    [2472, "ClimateAwareGNN", 46, 0.9136, 0.8914, 0.8667, 0.8125, 0.8387],
]


columns = [
    "network_id",
    "model",
    "seed",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
]


df = pd.DataFrame(
    results,
    columns=columns
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

raw_file = (
    OUTPUT_DIR
    / "climate_gnn_results_5seeds.csv"
)

df.to_csv(
    raw_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

metrics = [
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
]

summary = (
    df.groupby(
        ["network_id", "model"]
    )[metrics]
    .agg(["mean", "std"])
    .reset_index()
)


# Flatten column names

summary.columns = [
    "_".join(
        column
    ).strip("_")
    for column in summary.columns
]


summary_file = (
    OUTPUT_DIR
    / "climate_gnn_results_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("CAMT-GNN | CLIMATE-AWARE GNN RESULTS SAVED")
print("=" * 70)

print("\nRaw results:")
print(raw_file)

print("\nSummary results:")
print(summary_file)

print("\nSummary:")
print(
    summary.to_string(index=False)
)

print("\n" + "=" * 70)
print("SUCCESS — RESULTS SAVED")
print("=" * 70)