
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "hospital_feature_engineered.csv"
)

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("HOSPITAL MODEL VALIDATION")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nTarget distribution:")
print(df["Outcome"].value_counts(dropna=False))

print("\nOutcome by Satisfaction:")
print(pd.crosstab(df["Satisfaction"], df["Outcome"]))

for col in ["Length_of_Stay", "Total_Cost", "Readmission",
            "Insurance_Claimed", "Patient_State"]:
    if col in df.columns:
        print(f"\nOutcome by {col}:")
        print(pd.crosstab(df[col], df["Outcome"]))

print("\nDate and post-outcome columns to review:")
for col in ["Admission_Date", "Discharge_Date", "Length_of_Stay",
            "Satisfaction", "Total_Cost", "Readmission"]:
    if col in df.columns:
        print("-", col)

print("\nValidation check completed.")
