import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

# ============================================================
# PEARSON CORRELATION SIGNIFICANCE TEST
# ============================================================

BASE_DIR = Path(r"D:\Healthcare-BI-Dashboard")

DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "outputs" / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Load dataset
df = pd.read_csv(
    DATA_DIR / "obesity_dataset.csv"
)

# Select variables
height = df["Height"]
weight = df["Weight"]

# Pearson correlation test
correlation, p_value = pearsonr(
    height,
    weight
)

print("=" * 70)
print("PEARSON CORRELATION SIGNIFICANCE TEST")
print("=" * 70)

print(f"\nVariables: Height and Weight")
print(f"Pearson Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.6f}")

# Interpretation
if p_value < 0.05:
    result = "Statistically significant relationship"
else:
    result = "No statistically significant relationship"

print(f"\nResult: {result}")

# Save result
result_df = pd.DataFrame({
    "Variable_1": ["Height"],
    "Variable_2": ["Weight"],
    "Pearson_Correlation": [correlation],
    "P_Value": [p_value],
    "Significance": [result]
})

output_file = (
    RESULT_DIR /
    "height_weight_pearson_test.csv"
)

result_df.to_csv(
    output_file,
    index=False
)

print("\nResult saved to:")
print(output_file)

print("\n" + "=" * 70)
print("STATISTICAL TEST COMPLETED")
print("=" * 70)