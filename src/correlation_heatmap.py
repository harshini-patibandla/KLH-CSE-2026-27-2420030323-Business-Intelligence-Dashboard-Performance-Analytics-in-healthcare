import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CORRELATION HEATMAP
# ============================================================

BASE_DIR = Path(r"D:\Healthcare-BI-Dashboard")

DATA_DIR = BASE_DIR / "data"
GRAPH_DIR = BASE_DIR / "outputs" / "graphs"

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Load dataset
df = pd.read_csv(
    DATA_DIR / "obesity_dataset.csv"
)

# Select numerical columns
numeric_columns = df.select_dtypes(
    include="number"
).columns

# Correlation matrix
correlation = df[
    numeric_columns
].corr()

# Create heatmap
plt.figure(figsize=(10, 8))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

# Add correlation values
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title(
    "Correlation Heatmap of Obesity Dataset"
)

plt.tight_layout()

# Save graph
output_file = (
    GRAPH_DIR /
    "obesity_correlation_heatmap.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 70)
print("CORRELATION HEATMAP CREATED SUCCESSFULLY")
print("=" * 70)

print("\nSaved at:")
print(output_file)