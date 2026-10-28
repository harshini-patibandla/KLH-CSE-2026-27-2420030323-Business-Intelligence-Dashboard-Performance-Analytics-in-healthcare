import pandas as pd
from pathlib import Path

RESULT_DIR = Path("outputs/results")

baseline_file = RESULT_DIR / "obesity_model_results.csv"

# Change this to the actual filename containing your hybrid results
hybrid_file = RESULT_DIR / "obesity_hybrid_model_results.csv"

baseline = pd.read_csv(baseline_file)
hybrid = pd.read_csv(hybrid_file)

# Keep only the required columns
columns = [
    "Dataset", "Model", "Accuracy",
    "Precision", "Recall", "F1_Score"
]

hybrid = hybrid[columns]
baseline = baseline[columns]

# Keep the existing baseline rows and append hybrid rows
combined = pd.concat(
    [baseline, hybrid],
    ignore_index=True
)

# Prevent duplicate model rows if the script is run again
combined = combined.drop_duplicates(
    subset=["Dataset", "Model"],
    keep="last"
)

combined.to_csv(baseline_file, index=False)

print("Updated Obesity model results:")
print(combined.to_string(index=False))