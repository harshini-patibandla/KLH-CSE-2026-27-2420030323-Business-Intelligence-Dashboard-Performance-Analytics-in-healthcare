
# ============================================================
# FINAL MODEL COMPARISON - OBESITY
# BASELINE + 6 HYBRID MODELS
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"D:\Healthcare-BI-Dashboard"

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD BASELINE MODELS
# ============================================================

baseline_path = os.path.join(
    RESULTS_DIR,
    "obesity_final_model_comparison.csv"
)

print("\nLoading baseline models...")

if not os.path.exists(baseline_path):
    raise FileNotFoundError(
        f"Baseline file not found:\n{baseline_path}"
    )

baseline_raw = pd.read_csv(baseline_path)

print("Baseline columns:", baseline_raw.columns.tolist())


# ============================================================
# 3. RENAME BASELINE COLUMNS
# ============================================================

baseline_df = baseline_raw.rename(
    columns={
        "Model": "model",
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall / Sensitivity": "recall",
        "Recall": "recall",
        "F1-score": "f1",
        "F1 Score": "f1",
        "F1": "f1",
        "ROC-AUC": "roc_auc",
        "ROC_AUC": "roc_auc"
    }
)

required_columns = [
    "model", "accuracy", "precision", "recall", "f1", "roc_auc"
]

missing = [
    col for col in required_columns
    if col not in baseline_df.columns
]

if missing:
    raise ValueError(
        f"Missing baseline columns: {missing}\n"
        f"Available columns: {baseline_df.columns.tolist()}"
    )

baseline_df = baseline_df[required_columns].copy()


# ============================================================
# 4. HYBRID MODEL FILES
# ============================================================

hybrid_files = [
    (
        "Random Forest + MLP Hybrid",
        "obesity_hybrid_model_results.csv"
    ),
    (
        "Hybrid 2 - RF + MLP Feature Groups",
        "obesity_hybrid_candidate2_results.csv"
    ),
    (
        "Hybrid 3 - XGBoost + MLP",
        "obesity_hybrid_candidate3_results.csv"
    ),
    (
        "Hybrid 4 - RF to XGBoost",
        "obesity_hybrid_candidate4_results.csv"
    ),
    (
        "Hybrid 5 - Interaction + XGB + MLP",
        "obesity_hybrid_candidate5_results.csv"
    ),
    (
        "Hybrid 6 - Domain-Guided XGB + MLP + Meta-Learner",
        "obesity_hybrid_candidate6_results.csv"
    )
]


# ============================================================
# 5. READ HYBRID RESULT FILES
# ============================================================

def read_hybrid_file(file_path, model_name):

    df = pd.read_csv(file_path)

    if df.empty:
        print(f"WARNING: Empty file: {file_path}")
        return None

    # Normalize column names for reliable matching
    normalized = {
        str(col).strip().lower().replace("_", "-").replace(" ", ""): col
        for col in df.columns
    }

    def get_column(possible_names):
        for name in possible_names:
            key = name.lower().strip().replace("_", "-").replace(" ", "")
            if key in normalized:
                return normalized[key]
        return None

    accuracy_col = get_column(["accuracy", "acc"])
    precision_col = get_column(["precision"])
    recall_col = get_column([
        "recall",
        "recall/sensitivity",
        "recall / sensitivity",
        "sensitivity"
    ])
    f1_col = get_column([
        "f1",
        "f1-score",
        "f1 score",
        "f1_score"
    ])
    roc_col = get_column([
        "roc-auc",
        "roc_auc",
        "roc auc",
        "roc_auc_score"
    ])

    required = {
        "Accuracy": accuracy_col,
        "Precision": precision_col,
        "Recall": recall_col,
        "F1": f1_col,
        "ROC-AUC": roc_col
    }

    missing_cols = [
        name for name, col in required.items()
        if col is None
    ]

    if missing_cols:
        print(
            f"WARNING: {model_name} missing columns: {missing_cols}"
        )
        return None

    row = df.iloc[0]

    try:
        return {
            "model": model_name,
            "accuracy": float(row[accuracy_col]),
            "precision": float(row[precision_col]),
            "recall": float(row[recall_col]),
            "f1": float(row[f1_col]),
            "roc_auc": float(row[roc_col])
        }
    except (ValueError, TypeError):
        print(f"WARNING: Invalid metric value in {model_name}")
        return None


# ============================================================
# 6. LOAD ALL HYBRIDS
# ============================================================

hybrid_rows = []

for model_name, filename in hybrid_files:

    file_path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {filename}")
        continue

    result = read_hybrid_file(file_path, model_name)

    if result is not None:
        hybrid_rows.append(result)


# ============================================================
# 7. COMBINE BASELINE + HYBRID RESULTS
# ============================================================

hybrid_df = pd.DataFrame(hybrid_rows)

final_df = pd.concat(
    [baseline_df, hybrid_df],
    ignore_index=True
)

final_df = final_df.drop_duplicates(
    subset=["model"],
    keep="first"
).reset_index(drop=True)

# Ensure all metrics are numeric
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

for metric in metrics:
    final_df[metric] = pd.to_numeric(
        final_df[metric],
        errors="coerce"
    )


# ============================================================
# 8. DISPLAY AND SAVE COMPARISON CSV
# ============================================================

print("\n" + "=" * 90)
print("FINAL OBESITY MODEL COMPARISON")
print("=" * 90)
print(final_df.to_string(index=False))
print("\nTotal models included:", len(final_df))

csv_path = os.path.join(
    RESULTS_DIR,
    "obesity_final_12_model_comparison.csv"
)

final_df.to_csv(csv_path, index=False)
print("\nComparison CSV saved:", csv_path)


# ============================================================
# 9. FIND BEST HYBRID USING F1-SCORE
# ============================================================

# Only select actual hybrid models, not baseline models
hybrid_mask = final_df["model"].str.contains(
    "hybrid", case=False, na=False
)

hybrid_only = final_df[
    hybrid_mask & final_df["f1"].notna()
].copy()

best_hybrid = None

if not hybrid_only.empty:

    best_hybrid = hybrid_only.loc[
        hybrid_only["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST OBESITY HYBRID MODEL (BY F1-SCORE)")
    print("=" * 70)
    print("Model:", best_hybrid["model"])
    print(f"Accuracy:  {best_hybrid['accuracy']:.4f}")
    print(f"Precision: {best_hybrid['precision']:.4f}")
    print(f"Recall:    {best_hybrid['recall']:.4f}")
    print(f"F1-score:  {best_hybrid['f1']:.4f}")
    print(f"ROC-AUC:   {best_hybrid['roc_auc']:.4f}")

else:
    print("\nNo valid hybrid F1-scores were found.")


# ============================================================
# 10. GRAPH SETTINGS
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall / Sensitivity",
    "F1 Score",
    "ROC-AUC"
]

models = final_df["model"].tolist()
x = np.arange(len(models))
width = 0.15

fig, ax = plt.subplots(figsize=(24, 11))


# ============================================================
# 11. DRAW METRIC BARS
# ============================================================

for i, metric in enumerate(metrics):

    values = final_df[metric].values

    positions = x + (i - 2) * width

    bars = ax.bar(
        positions,
        values,
        width,
        label=metric_names[i]
    )

    for bar, value in zip(bars, values):

        if pd.isna(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90
        )


# ============================================================
# 12. HIGHLIGHT BEST HYBRID ABOVE ITS F1 BAR
# ============================================================

if best_hybrid is not None:

    best_model_name = best_hybrid["model"]
    best_f1 = float(best_hybrid["f1"])

    best_index = final_df.index[
        final_df["model"] == best_model_name
    ][0]

    # F1 is the fourth metric, index 3
    f1_bar_x = x[best_index] + (3 - 2) * width

    ax.annotate(
        f"Best Hybrid\nF1 = {best_f1:.4f}",
        xy=(f1_bar_x, best_f1),
        xytext=(f1_bar_x, best_f1 + 0.035),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.2
        }
    )


# ============================================================
# 13. AXIS LABELS AND TITLE
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(
    models,
    rotation=45,
    ha="right",
    fontsize=9
)

ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Performance Score", fontsize=12)

ax.set_title(
    "Comparative Analysis of Baseline and Hybrid Models - Obesity",
    fontsize=16,
    fontweight="bold"
)


# ============================================================
# 14. Y-AXIS, LEGEND AND GRID
# ============================================================

# Full scale so lower-performing values are not cut off
ax.set_ylim(0, 1.10)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=5
)

ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)


# ============================================================
# 15. SAVE GRAPH
# ============================================================

plt.tight_layout()

graph_path = os.path.join(
    GRAPHS_DIR,
    "obesity_final_12_model_comparison.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 90)
print("COMPLETED SUCCESSFULLY")
print("=" * 90)

print("\nFinal CSV:")
print(csv_path)

print("\nFinal graph:")
print(graph_path)

print("\nModels included:")
for model in models:
    print(" -", model)

print("\nMetrics included:", ", ".join(metric_names))
