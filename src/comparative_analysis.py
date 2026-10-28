# ============================================================
# COMPARATIVE ANALYSIS
# Healthcare BI Dashboard
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPHS_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD EVALUATION RESULTS
# ============================================================

input_file = os.path.join(
    RESULTS_DIR,
    "all_evaluation_metrics.csv"
)

df = pd.read_csv(
    input_file
)

print("\nEvaluation results loaded successfully.")

print(
    "\nDataset shape:",
    df.shape
)


# ============================================================
# 3. DISPLAY COMPLETE COMPARISON TABLE
# ============================================================

print("\n" + "=" * 120)

print(
    "COMPLETE MODEL COMPARISON"
)

print("=" * 120)

print(
    df.round(4).to_string(
        index=False
    )
)


# ============================================================
# 4. SAVE COMPARATIVE TABLE
# ============================================================

comparison_file = os.path.join(
    RESULTS_DIR,
    "comparative_analysis_table.csv"
)

df.to_csv(
    comparison_file,
    index=False
)

print(
    "\nSaved comparative table:"
)

print(
    comparison_file
)


# ============================================================
# 5. OVERALL MODEL PERFORMANCE
# ============================================================

overall_model_performance = (
    df.groupby("Model")[
        [
            "Accuracy",
            "Precision",
            "Recall / Sensitivity",
            "F1-score",
            "ROC-AUC"
        ]
    ]
    .mean()
    .reset_index()
)


print("\n" + "=" * 100)

print(
    "AVERAGE PERFORMANCE ACROSS ALL THREE DATASETS"
)

print("=" * 100)

print(
    overall_model_performance.round(4).to_string(
        index=False
    )
)


# ============================================================
# 6. SAVE OVERALL MODEL PERFORMANCE
# ============================================================

overall_file = os.path.join(
    RESULTS_DIR,
    "overall_model_performance.csv"
)

overall_model_performance.to_csv(
    overall_file,
    index=False
)

print(
    "\nSaved overall model performance:"
)

print(
    overall_file
)


# ============================================================
# 7. GRAPH 1 — MODEL PERFORMANCE COMPARISON
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall / Sensitivity",
    "F1-score",
    "ROC-AUC"
]

models = overall_model_performance[
    "Model"
].tolist()

x = range(
    len(models)
)

width = 0.15

plt.figure(
    figsize=(14, 7)
)

for i, metric in enumerate(metrics):

    values = overall_model_performance[
        metric
    ]

    positions = [
        value + (i - 2) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )


plt.xticks(
    list(x),
    models,
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.title(
    "Overall Model Performance Comparison"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


overall_graph = os.path.join(
    GRAPHS_DIR,
    "overall_model_performance_comparison.png"
)

plt.savefig(
    overall_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved graph:"
)

print(
    overall_graph
)


# ============================================================
# 8. GRAPH 2 — DATASET-WISE ACCURACY
# ============================================================

pivot_accuracy = df.pivot(
    index="Model",
    columns="Dataset",
    values="Accuracy"
)

plt.figure(
    figsize=(12, 7)
)

pivot_accuracy.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.ylabel(
    "Accuracy"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.title(
    "Accuracy Comparison Across Healthcare Datasets"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.legend(
    title="Dataset"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


accuracy_graph = os.path.join(
    GRAPHS_DIR,
    "dataset_wise_accuracy_comparison.png"
)

plt.savefig(
    accuracy_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved graph:"
)

print(
    accuracy_graph
)


# ============================================================
# 9. DATASET-WISE SUMMARY
# ============================================================

dataset_summary = (
    df.groupby("Dataset")[
        [
            "Accuracy",
            "Precision",
            "Recall / Sensitivity",
            "F1-score",
            "ROC-AUC"
        ]
    ]
    .mean()
    .reset_index()
)


print("\n" + "=" * 100)

print(
    "AVERAGE PERFORMANCE BY DATASET"
)

print("=" * 100)

print(
    dataset_summary.round(4).to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE DATASET SUMMARY
# ============================================================

dataset_summary_file = os.path.join(
    RESULTS_DIR,
    "dataset_performance_summary.csv"
)

dataset_summary.to_csv(
    dataset_summary_file,
    index=False
)

print(
    "\nSaved dataset summary:"
)

print(
    dataset_summary_file
)


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 100)

print(
    "COMPARATIVE ANALYSIS COMPLETED SUCCESSFULLY"
)

print("=" * 100)

print(
    "\nResults saved in:"
)

print(
    RESULTS_DIR
)

print(
    "\nGraphs saved in:"
)

print(
    GRAPHS_DIR
)