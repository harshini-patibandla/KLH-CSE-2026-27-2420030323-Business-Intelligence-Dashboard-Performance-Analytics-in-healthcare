# ============================================================
# FEATURE ENGINEERING
# Healthcare BI Dashboard
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CLEANED_DATA_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned_data"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD CLEANED DATASETS
# ============================================================

obesity_path = os.path.join(
    CLEANED_DATA_DIR,
    "obesity_cleaned.csv"
)

hospital_path = os.path.join(
    CLEANED_DATA_DIR,
    "hospital_cleaned.csv"
)

healthcare_path = os.path.join(
    CLEANED_DATA_DIR,
    "healthcare_cleaned.csv"
)


obesity_df = pd.read_csv(obesity_path)
hospital_df = pd.read_csv(hospital_path)
healthcare_df = pd.read_csv(healthcare_path)


print("=" * 60)
print("CLEANED DATASETS LOADED")
print("=" * 60)

print("Obesity:", obesity_df.shape)
print("Hospital:", hospital_df.shape)
print("Healthcare:", healthcare_df.shape)


# ============================================================
# 3. OBESITY FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("OBESITY FEATURE ENGINEERING")
print("=" * 60)


# BMI already exists from preprocessing.
# Create BMI category.

if "BMI" in obesity_df.columns:

    obesity_df["BMI_Category"] = pd.cut(
        obesity_df["BMI"],
        bins=[
            0,
            18.5,
            25,
            30,
            float("inf")
        ],
        labels=[
            "Underweight",
            "Normal",
            "Overweight",
            "Obese"
        ],
        include_lowest=True
    )

    print("BMI_Category created.")


# Create age groups

if "Age" in obesity_df.columns:

    obesity_df["Age_Group"] = pd.cut(
        obesity_df["Age"],
        bins=[
            0,
            18,
            30,
            45,
            60,
            float("inf")
        ],
        labels=[
            "Under 18",
            "18-30",
            "31-45",
            "46-60",
            "60+"
        ],
        include_lowest=True
    )

    print("Age_Group created.")


# Create physical activity category

if "FAF" in obesity_df.columns:

    obesity_df["Physical_Activity_Level"] = pd.cut(
        obesity_df["FAF"],
        bins=[
            -float("inf"),
            1,
            2,
            float("inf")
        ],
        labels=[
            "Low",
            "Moderate",
            "High"
        ]
    )

    print("Physical_Activity_Level created.")


# Create water intake category

if "CH2O" in obesity_df.columns:

    obesity_df["Water_Intake_Level"] = pd.cut(
        obesity_df["CH2O"],
        bins=[
            -float("inf"),
            1,
            2,
            float("inf")
        ],
        labels=[
            "Low",
            "Moderate",
            "High"
        ]
    )

    print("Water_Intake_Level created.")


# ============================================================
# 4. HOSPITAL FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("HOSPITAL FEATURE ENGINEERING")
print("=" * 60)


# Convert dates again after loading CSV

hospital_df["Admission_Date"] = pd.to_datetime(
    hospital_df["Admission_Date"],
    errors="coerce"
)

hospital_df["Discharge_Date"] = pd.to_datetime(
    hospital_df["Discharge_Date"],
    errors="coerce"
)


# Length of stay already exists.
# Create stay category.

if "Length_of_Stay" in hospital_df.columns:

    hospital_df["Stay_Category"] = pd.cut(
        hospital_df["Length_of_Stay"],
        bins=[
            -1,
            3,
            7,
            float("inf")
        ],
        labels=[
            "Short Stay",
            "Medium Stay",
            "Long Stay"
        ]
    )

    print("Stay_Category created.")


# Create age groups

if "Age" in hospital_df.columns:

    hospital_df["Age_Group"] = pd.cut(
        hospital_df["Age"],
        bins=[
            0,
            18,
            30,
            45,
            60,
            float("inf")
        ],
        labels=[
            "Under 18",
            "18-30",
            "31-45",
            "46-60",
            "60+"
        ],
        include_lowest=True
    )

    print("Age_Group created.")


# Create admission year

if "Admission_Date" in hospital_df.columns:

    hospital_df["Admission_Year"] = (
        hospital_df["Admission_Date"].dt.year
    )

    print("Admission_Year created.")


# Create admission month

if "Admission_Date" in hospital_df.columns:

    hospital_df["Admission_Month"] = (
        hospital_df["Admission_Date"].dt.month
    )

    print("Admission_Month created.")


# ============================================================
# 5. HEALTHCARE FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("HEALTHCARE FEATURE ENGINEERING")
print("=" * 60)


# Convert dates

healthcare_df["Date of Admission"] = pd.to_datetime(
    healthcare_df["Date of Admission"],
    errors="coerce"
)

healthcare_df["Discharge Date"] = pd.to_datetime(
    healthcare_df["Discharge Date"],
    errors="coerce"
)


# Length of stay already exists.
# Create stay category.

if "Length_of_Stay" in healthcare_df.columns:

    healthcare_df["Stay_Category"] = pd.cut(
        healthcare_df["Length_of_Stay"],
        bins=[
            -1,
            3,
            7,
            float("inf")
        ],
        labels=[
            "Short Stay",
            "Medium Stay",
            "Long Stay"
        ]
    )

    print("Stay_Category created.")


# Create age groups

if "Age" in healthcare_df.columns:

    healthcare_df["Age_Group"] = pd.cut(
        healthcare_df["Age"],
        bins=[
            0,
            18,
            30,
            45,
            60,
            float("inf")
        ],
        labels=[
            "Under 18",
            "18-30",
            "31-45",
            "46-60",
            "60+"
        ],
        include_lowest=True
    )

    print("Age_Group created.")


# Create admission year

if "Date of Admission" in healthcare_df.columns:

    healthcare_df["Admission_Year"] = (
        healthcare_df["Date of Admission"].dt.year
    )

    print("Admission_Year created.")


# Create admission month

if "Date of Admission" in healthcare_df.columns:

    healthcare_df["Admission_Month"] = (
        healthcare_df["Date of Admission"].dt.month
    )

    print("Admission_Month created.")


# Create billing amount category

if "Billing Amount" in healthcare_df.columns:

    healthcare_df["Billing_Category"] = pd.qcut(
        healthcare_df["Billing Amount"],
        q=3,
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        duplicates="drop"
    )

    print("Billing_Category created.")


# ============================================================
# 6. DISPLAY NEW FEATURES
# ============================================================

print("\n" + "=" * 60)
print("NEW FEATURES CREATED")
print("=" * 60)


print("\nObesity columns:")
print(obesity_df.columns.tolist())


print("\nHospital columns:")
print(hospital_df.columns.tolist())


print("\nHealthcare columns:")
print(healthcare_df.columns.tolist())


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING DATA QUALITY CHECK")
print("=" * 60)


print(
    "Obesity missing values:",
    obesity_df.isnull().sum().sum()
)

print(
    "Hospital missing values:",
    hospital_df.isnull().sum().sum()
)

print(
    "Healthcare missing values:",
    healthcare_df.isnull().sum().sum()
)


# ============================================================
# 8. SAVE FEATURE-ENGINEERED DATASETS
# ============================================================

obesity_output = os.path.join(
    RESULTS_DIR,
    "obesity_feature_engineered.csv"
)

hospital_output = os.path.join(
    RESULTS_DIR,
    "hospital_feature_engineered.csv"
)

healthcare_output = os.path.join(
    RESULTS_DIR,
    "healthcare_feature_engineered.csv"
)


obesity_df.to_csv(
    obesity_output,
    index=False
)

hospital_df.to_csv(
    hospital_output,
    index=False
)

healthcare_df.to_csv(
    healthcare_output,
    index=False
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFinal shapes:")

print(
    "Obesity:",
    obesity_df.shape
)

print(
    "Hospital:",
    hospital_df.shape
)

print(
    "Healthcare:",
    healthcare_df.shape
)

print("\nFiles saved to:")

print(obesity_output)
print(hospital_output)
print(healthcare_output)

print("\nAll feature engineering steps completed!")