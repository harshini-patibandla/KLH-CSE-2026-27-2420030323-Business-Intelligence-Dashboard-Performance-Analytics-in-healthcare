
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Project folder
ROOT = Path(__file__).resolve().parent.parent

csv_path = ROOT / "outputs" / "results" / "obesity_final_12_model_comparison.csv"
output_path = ROOT / "outputs" / "graphs" / "obesity_final_12_model_comparison.png"

# Load actual saved results
df = pd.read_csv(csv_path)

metrics = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall / Sensitivity",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}

# Convert metric values to numeric
for column in metrics:
    df[column] = pd.to_numeric(df[column], errors="raise")

# Check expected number of models
if len(df) != 12:
    raise ValueError(
        f"Expected 12 models, but found {len(df)} rows."
    )

x = np.arange(len(df))
width = 0.15

fig, ax = plt.subplots(figsize=(20, 9))

for i, (column, label) in enumerate(metrics.items()):
    positions = x + (i - 2) * width
    bars = ax.bar(
        positions,
        df[column],
        width,
        label=label
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        fontsize=6,
        padding=2,
        rotation=90
    )

ax.set_title(
    "Comparative Analysis of 12 Baseline and Hybrid Models",
    fontsize=16,
    fontweight="bold"
)
ax.set_xlabel("Machine Learning Models")
ax.set_ylabel("Performance Score")
ax.set_xticks(x)
ax.set_xticklabels(
    df["model"],
    rotation=35,
    ha="right",
    fontsize=8
)
ax.set_ylim(0.90, 1.025)
ax.legend(ncol=5, loc="upper center")
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Graph created successfully:")
print(output_path)
print(f"Models included: {len(df)}")