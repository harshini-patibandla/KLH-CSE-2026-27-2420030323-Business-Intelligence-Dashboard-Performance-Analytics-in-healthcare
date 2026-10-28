
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results for all 8 models
input_path = "outputs/results/hospital_all_models_comparison.csv"
output_path = "outputs/graphs/hospital_all_models_comparison.png"

os.makedirs("outputs/graphs", exist_ok=True)

df = pd.read_csv(input_path)

metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]

if len(df) != 8:
    raise ValueError(f"Expected 8 models, but found {len(df)}")

x = np.arange(len(metrics))
width = 0.10

fig, ax = plt.subplots(figsize=(16, 8))

for i, (_, row) in enumerate(df.iterrows()):
    values = [row[metric] for metric in metrics]
    positions = x + (i - (len(df) - 1) / 2) * width

    bars = ax.bar(
        positions,
        values,
        width,
        label=row["Model"]
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.3f}" for v in values],
        padding=2,
        fontsize=7,
        rotation=90
    )

ax.set_title(
    "Hospital Baseline and Hybrid Models - Performance Comparison",
    fontsize=15
)
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15)

ax.legend(
    title="Models",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9
)

ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Graph generated successfully.")
print("Models included:", len(df))
print("Saved to:", output_path)
print(df[["Model", "Accuracy", "Precision", "Recall", "F1_Score"]].to_string(index=False))
