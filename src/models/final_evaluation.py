from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data/results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS / "final_model_comparison.csv")

metrics = ["roc_auc", "average_precision", "precision", "recall", "f1"]

summary = (
    df.groupby("model")[metrics]
    .mean()
    .sort_values("roc_auc", ascending=False)
)

summary.to_csv(RESULTS / "final_model_summary.csv")

for metric in metrics:
    plt.figure(figsize=(9, 5))
    summary[metric].sort_values().plot(kind="barh")
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Model")
    plt.title(f"CAMT-GNN Model Comparison — {metric.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{metric}_comparison.png", dpi=300)
    plt.close()

print("Final evaluation completed.")
print("\nModel summary:")
print(summary.round(4).to_string())

print(f"\nSaved: {RESULTS / 'final_model_summary.csv'}")
print(f"Figures saved: {FIGURES}")