# ============================================================
# STATISTICAL ANALYSIS — HEALTHCARE BI DASHBOARD
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(r"D:\Healthcare-BI-Dashboard")

DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "outputs" / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD OBESITY DATASET
# ============================================================

file_path = DATA_DIR / "obesity_dataset.csv"

df = pd.read_csv(file_path)

print("=" * 70)
print("STATISTICAL ANALYSIS — OBESITY DATASET")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNumerical Features:")
print(numeric_columns)

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

statistics = df[numeric_columns].describe().T

statistics["median"] = df[
    numeric_columns
].median()

statistics["variance"] = df[
    numeric_columns
].var()

statistics = statistics[
    [
        "count",
        "mean",
        "median",
        "std",
        "variance",
        "min",
        "max"
    ]
]

statistics = statistics.round(4)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nDescriptive Statistics:")
print(statistics)

# ============================================================
# SAVE STATISTICAL SUMMARY
# ============================================================

output_file = (
    RESULT_DIR /
    "obesity_statistical_summary.csv"
)

statistics.to_csv(
    output_file
)

print("\nStatistical summary saved to:")

print(output_file)

# ============================================================
# CORRELATION MATRIX
# ============================================================

correlation = df[
    numeric_columns
].corr()

correlation = correlation.round(4)

print("\nCorrelation Matrix:")
print(correlation)

# ============================================================
# SAVE CORRELATION MATRIX
# ============================================================

correlation_file = (
    RESULT_DIR /
    "obesity_correlation_matrix.csv"
)

correlation.to_csv(
    correlation_file
)

print("\nCorrelation matrix saved to:")

print(correlation_file)

# ============================================================
# STRONGEST CORRELATIONS
# ============================================================

correlation_pairs = (
    correlation
    .where(
        np.triu(
            np.ones(
                correlation.shape
            ),
            k=1
        ).astype(bool)
    )
    .stack()
    .reset_index()
)

correlation_pairs.columns = [
    "Feature_1",
    "Feature_2",
    "Correlation"
]

correlation_pairs[
    "Absolute_Correlation"
] = correlation_pairs[
    "Correlation"
].abs()

correlation_pairs = (
    correlation_pairs
    .sort_values(
        "Absolute_Correlation",
        ascending=False
    )
)

print("\nTop Correlations:")

print(
    correlation_pairs.head(10)
)

# ============================================================
# SAVE TOP CORRELATIONS
# ============================================================

correlation_pairs_file = (
    RESULT_DIR /
    "obesity_top_correlations.csv"
)

correlation_pairs.to_csv(
    correlation_pairs_file,
    index=False
)

print("\nTop correlations saved to:")

print(correlation_pairs_file)

# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)