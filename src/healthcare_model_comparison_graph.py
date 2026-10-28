
# ============================================================
# HEALTHCARE DATASET
# BASELINE + HYBRID MODEL PERFORMANCE COMPARISON
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. Define input and output paths
# ------------------------------------------------------------

input_path = "outputs/results/healthcare_all_models_comparison.csv"
output_path = "outputs/graphs/healthcare_all_models_comparison.png"

os.makedirs("outputs/graphs", exist_ok=True)

# ------------------------------------------------------------
# 2. Load comparison results
# ------------------------------------------------------------

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"Comparison CSV not found: {input_path}"
    )

df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# 3. Validate the data
# ------------------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
required_columns = ["Model"] + metrics

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if len(df) != 8:
    raise ValueError(
        f"Expected 8 models, found {len(df)}"
    )

for metric in metrics:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

if df[metrics].isna().any().any():
    raise ValueError(
        "Some performance metrics are missing or non-numeric."
    )

# ------------------------------------------------------------
# 4. Identify the best hybrid by F1-score
# ------------------------------------------------------------


hybrid_mask = df["Model"].astype(str).str.contains(
    r"hybrid|probability\s*fusion",
    case=False,
    na=False,
    regex=True
)


hybrid_models = df[hybrid_mask]

if hybrid_models.empty:
    raise ValueError(
        "No hybrid models found. Check the Model names in the CSV."
    )

best_hybrid_index = hybrid_models["F1_Score"].idxmax()
best_hybrid = df.loc[best_hybrid_index]

best_hybrid_name = best_hybrid["Model"]
best_hybrid_f1 = best_hybrid["F1_Score"]

# ------------------------------------------------------------
# 5. Create grouped bar chart
# ------------------------------------------------------------

x = np.arange(len(metrics))
number_of_models = len(df)
width = 0.10

fig, ax = plt.subplots(figsize=(16, 8))

for i, (_, row) in enumerate(df.iterrows()):

    values = [row[metric] for metric in metrics]

    positions = x + (
        i - (number_of_models - 1) / 2
    ) * width

    bars = ax.bar(
        positions,
        values,
        width,
        label=row["Model"]
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in values],
        padding=2,
        fontsize=7,
        rotation=90
    )

# ------------------------------------------------------------
# 6. Add title and formatting
# ------------------------------------------------------------

ax.set_title(
    "Healthcare Baseline and Hybrid Models - Performance Comparison",
    fontsize=15,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Evaluation Metrics", fontsize=12)
ax.set_ylabel("Performance Score", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)

ax.set_ylim(0, 0.65)

ax.legend(
    title="Models",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

# ------------------------------------------------------------
# 7. Highlight the best hybrid model
# ------------------------------------------------------------

ax.text(
    0.5,
    1.02,
    f"BEST HYBRID: {best_hybrid_name}\n"
    f"F1-Score = {best_hybrid_f1:.4f}",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black"
    )
)

# Leave space for the label and legend
fig.subplots_adjust(
    top=0.78,
    right=0.78,
    bottom=0.12
)

# ------------------------------------------------------------
# 8. Save the graph
# ------------------------------------------------------------

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 9. Print confirmation
# ------------------------------------------------------------

print("=" * 65)
print("HEALTHCARE MODEL COMPARISON GRAPH CREATED SUCCESSFULLY")
print("=" * 65)

print("Models included:", len(df))
print("Best hybrid model:", best_hybrid_name)
print(f"Best hybrid F1-score: {best_hybrid_f1:.4f}")
print("Saved to:", output_path)

print("\nMODEL PERFORMANCE RESULTS")
print("-" * 65)

print(
    df[["Model"] + metrics].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)
