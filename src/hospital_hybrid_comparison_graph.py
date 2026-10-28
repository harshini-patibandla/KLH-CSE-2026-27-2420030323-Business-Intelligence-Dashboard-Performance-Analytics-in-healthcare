
# ============================================================
# HOSPITAL DATASET
# BASELINE + HYBRID MODEL PERFORMANCE COMPARISON
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. Define input and output paths
# ------------------------------------------------------------

input_path = "outputs/results/hospital_all_models_comparison.csv"
output_path = "outputs/graphs/hospital_all_models_comparison.png"

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
# 3. Validate required columns
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
        f"Expected 8 models, but found {len(df)}."
    )

if df["Model"].isna().any():
    raise ValueError("Some model names are missing.")

for metric in metrics:
    df[metric] = pd.to_numeric(
        df[metric], errors="coerce"
    )

if df[metrics].isna().any().any():
    raise ValueError(
        "Some metric values are missing or non-numeric."
    )

# ------------------------------------------------------------
# 4. Identify the best hybrid model(s)
# ------------------------------------------------------------

hybrid_mask = df["Model"].astype(str).str.contains(
    r"probability\s*fusion|hybrid",
    case=False,
    na=False,
    regex=True
)

hybrid_df = df[hybrid_mask].copy()

if hybrid_df.empty:
    raise ValueError(
        "No hybrid models found in the comparison CSV."
    )

best_hybrid_f1 = hybrid_df["F1_Score"].max()

best_hybrids = hybrid_df[
    np.isclose(
        hybrid_df["F1_Score"],
        best_hybrid_f1,
        atol=1e-8
    )
]

best_hybrid_names = " / ".join(
    best_hybrids["Model"].astype(str).tolist()
)

annotation = (
    "BEST HYBRID MODEL(S)\n"
    f"{best_hybrid_names}\n"
    f"Best hybrid F1-Score = {best_hybrid_f1:.4f}"
)

# ------------------------------------------------------------
# 5. Create grouped bar chart
# ------------------------------------------------------------

x = np.arange(len(metrics))
number_of_models = len(df)
width = 0.10

fig, ax = plt.subplots(figsize=(17, 9))

for i, (_, row) in enumerate(df.iterrows()):

    values = [row[metric] for metric in metrics]

    positions = x + (
        i - (number_of_models - 1) / 2
    ) * width

    bars = ax.bar(
        positions,
        values,
        width,
        label=str(row["Model"])
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in values],
        padding=2,
        fontsize=7,
        rotation=90
    )

# ------------------------------------------------------------
# 6. Chart title and axis labels
# ------------------------------------------------------------

ax.set_title(
    "Hospital Baseline and Hybrid Model Performance Comparison",
    fontsize=15,
    fontweight="bold",
    pad=14
)

ax.set_xlabel("Evaluation Metrics", fontsize=12)
ax.set_ylabel("Performance Score", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)

ax.set_ylim(0, 1.18)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.35
)

ax.set_axisbelow(True)

# ------------------------------------------------------------
# 7. Place annotation INSIDE the top of the graph
# ------------------------------------------------------------

ax.text(
    0.50,
    0.98,
    annotation,
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=9,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black",
        alpha=0.95
    ),
    zorder=10
)

# ------------------------------------------------------------
# 8. Display model legend
# ------------------------------------------------------------

ax.legend(
    title="Models",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9
)

# ------------------------------------------------------------
# 9. Adjust layout
# ------------------------------------------------------------

fig.subplots_adjust(
    left=0.07,
    right=0.79,
    bottom=0.12,
    top=0.88
)

# ------------------------------------------------------------
# 10. Save graph
# ------------------------------------------------------------

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# 11. Print results
# ------------------------------------------------------------

print("=" * 70)
print("HOSPITAL MODEL COMPARISON GRAPH GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\nModels included: {len(df)}")
print(f"Hybrid models found: {len(hybrid_df)}")
print(f"Graph saved to: {output_path}")

print("\nBEST HYBRID MODEL(S)")
print("-" * 70)

print(
    best_hybrids[["Model"] + metrics].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nALL MODEL PERFORMANCE RESULTS")
print("-" * 70)

print(
    df[["Model"] + metrics].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\n" + "=" * 70)
