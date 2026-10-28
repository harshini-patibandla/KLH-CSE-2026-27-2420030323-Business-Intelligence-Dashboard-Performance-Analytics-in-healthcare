# ============================================================
# FINAL MODEL COMPARISON
# Existing Models + Proposed RF + MLP Hybrid
# Dataset: Obesity
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results"
)

GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "graphs"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# ============================================================
# 2. FILE PATHS
# ============================================================

existing_file = os.path.join(
    RESULTS_DIR,
    "comparative_analysis_table.csv"
)

hybrid_file = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_model_results.csv"
)


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 100)
print("FINAL MODEL COMPARISON")
print("Existing Models + Proposed RF + MLP Hybrid")
print("=" * 100)

if not os.path.exists(existing_file):
    raise FileNotFoundError(
        f"Existing model comparison file not found:\n{existing_file}"
    )

if not os.path.exists(hybrid_file):
    raise FileNotFoundError(
        f"Hybrid model result file not found:\n{hybrid_file}"
    )


# ============================================================
# 4. LOAD EXISTING MODEL RESULTS
# ============================================================

existing_df = pd.read_csv(existing_file)

# Keep only Obesity dataset
obesity_existing = existing_df[
    existing_df["Dataset"].str.lower() == "obesity"
].copy()

print("\nExisting Obesity model results:")
print(obesity_existing.to_string(index=False))


# ============================================================
# 5. LOAD HYBRID RESULT
# ============================================================

hybrid_df = pd.read_csv(hybrid_file)

print("\nProposed Hybrid model result:")
print(hybrid_df.to_string(index=False))


# ============================================================
# 6. COMBINE RESULTS
# ============================================================

final_df = pd.concat(
    [
        obesity_existing,
        hybrid_df
    ],
    ignore_index=True
)


# ============================================================
# 7. KEEP REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Dataset",
    "Model",
    "Accuracy",
    "Precision",
    "Recall / Sensitivity",
    "F1-score",
    "ROC-AUC"
]

final_df = final_df[required_columns]


# ============================================================
# 8. DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 100)
print("OBESITY — FINAL MODEL COMPARISON")
print("=" * 100)

print(
    final_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. SAVE FINAL COMPARISON TABLE
# ============================================================

comparison_path = os.path.join(
    RESULTS_DIR,
    "obesity_final_model_comparison.csv"
)

final_df.to_csv(
    comparison_path,
    index=False
)

print("\nFinal comparison table saved to:")
print(comparison_path)


# ============================================================
# 10. PREPARE GRAPH DATA
# ============================================================

plot_df = final_df.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall / Sensitivity",
        "F1-score",
        "ROC-AUC"
    ]
]


# ============================================================
# 11. CREATE COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(14, 7))

plot_df.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title(
    "Obesity Dataset - Existing Models vs Proposed Hybrid Model"
)

plt.xlabel("Model")
plt.ylabel("Score")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(0, 1.05)

plt.legend(
    title="Evaluation Metrics"
)

plt.tight_layout()


# ============================================================
# 12. SAVE GRAPH
# ============================================================

graph_path = os.path.join(
    GRAPHS_DIR,
    "obesity_final_model_comparison.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison graph saved to:")
print(graph_path)


# ============================================================
# 13. FIND BEST VALUE FOR EACH METRIC
# ============================================================

print("\n" + "=" * 100)
print("METRIC-WISE COMPARISON")
print("=" * 100)

metrics = [
    "Accuracy",
    "Precision",
    "Recall / Sensitivity",
    "F1-score",
    "ROC-AUC"
]

for metric in metrics:

    best_index = final_df[metric].idxmax()

    best_model = final_df.loc[
        best_index,
        "Model"
    ]

    best_value = final_df.loc[
        best_index,
        metric
    ]

    print(
        f"{metric:<25}: "
        f"{best_model:<30} "
        f"{best_value:.4f}"
    )


# ============================================================
# 14. HYBRID VS RANDOM FOREST
# ============================================================

print("\n" + "=" * 100)
print("PROPOSED HYBRID VS RANDOM FOREST")
print("=" * 100)

rf_row = final_df[
    final_df["Model"] == "Random Forest"
].iloc[0]

hybrid_row = final_df[
    final_df["Model"] == "Random Forest + MLP Hybrid"
].iloc[0]

for metric in metrics:

    rf_value = rf_row[metric]
    hybrid_value = hybrid_row[metric]

    difference = hybrid_value - rf_value

    print(
        f"{metric:<25}: "
        f"RF = {rf_value:.4f} | "
        f"Hybrid = {hybrid_value:.4f} | "
        f"Difference = {difference:+.4f}"
    )


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 100)

print("\nExisting models:")
print("1. Logistic Regression")
print("2. Decision Tree")
print("3. Random Forest")
print("4. XGBoost")
print("5. CatBoost")
print("6. MLP")

print("\nProposed hybrid model:")
print("7. Random Forest + MLP Probability Fusion")

print("\nEvaluation metrics:")
print("1. Accuracy")
print("2. Precision")
print("3. Recall / Sensitivity")
print("4. F1-score")
print("5. ROC-AUC")

print("\nFiles generated:")
print(comparison_path)
print(graph_path)