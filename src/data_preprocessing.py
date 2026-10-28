# ============================================================
# DATA PREPROCESSING
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

DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

obesity_path = os.path.join(
    DATA_DIR,
    "obesity_dataset.csv"
)

hospital_path = os.path.join(
    DATA_DIR,
    "Hospital_Patient.csv"
)

healthcare_path = os.path.join(
    DATA_DIR,
    "healthcare_dataset.csv"
)


obesity_df = pd.read_csv(obesity_path)
hospital_df = pd.read_csv(hospital_path)
healthcare_df = pd.read_csv(healthcare_path)


print("=" * 60)
print("DATASETS LOADED SUCCESSFULLY")
print("=" * 60)

print("Obesity Dataset Shape:", obesity_df.shape)
print("Hospital Dataset Shape:", hospital_df.shape)
print("Healthcare Dataset Shape:", healthcare_df.shape)


# ============================================================
# 3. CREATE WORKING COPIES
# ============================================================

obesity_clean = obesity_df.copy()
hospital_clean = hospital_df.copy()
healthcare_clean = healthcare_df.copy()


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

obesity_clean.columns = obesity_clean.columns.str.strip()
hospital_clean.columns = hospital_clean.columns.str.strip()
healthcare_clean.columns = healthcare_clean.columns.str.strip()

print("\nColumn names cleaned successfully.")


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

print(
    "Obesity duplicates:",
    obesity_clean.duplicated().sum()
)

print(
    "Hospital duplicates:",
    hospital_clean.duplicated().sum()
)

print(
    "Healthcare duplicates:",
    healthcare_clean.duplicated().sum()
)


obesity_clean = obesity_clean.drop_duplicates().copy()
hospital_clean = hospital_clean.drop_duplicates().copy()
healthcare_clean = healthcare_clean.drop_duplicates().copy()

print("\nDuplicates removed successfully.")


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print(
    "\nObesity missing values:",
    obesity_clean.isnull().sum().sum()
)

print(
    "Hospital missing values:",
    hospital_clean.isnull().sum().sum()
)

print(
    "Healthcare missing values:",
    healthcare_clean.isnull().sum().sum()
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numerical_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                df[column].median()
            )


    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:

        if df[column].isnull().sum() > 0:

            mode_value = df[column].mode()

            if not mode_value.empty:

                df[column] = df[column].fillna(
                    mode_value.iloc[0]
                )

    return df


obesity_clean = handle_missing_values(obesity_clean)
hospital_clean = handle_missing_values(hospital_clean)
healthcare_clean = handle_missing_values(healthcare_clean)

print("\nMissing values handled successfully.")


# ============================================================
# 8. CONVERT HOSPITAL DATES
# ============================================================

print("\n" + "=" * 60)
print("HOSPITAL DATE PROCESSING")
print("=" * 60)


hospital_clean["Admission_Date"] = pd.to_datetime(
    hospital_clean["Admission_Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

hospital_clean["Discharge_Date"] = pd.to_datetime(
    hospital_clean["Discharge_Date"],
    format="%d-%m-%Y",
    errors="coerce"
)


invalid_hospital_dates = (
    hospital_clean["Admission_Date"].isna()
    |
    hospital_clean["Discharge_Date"].isna()
).sum()


print(
    "Hospital records with invalid dates:",
    invalid_hospital_dates
)


# ============================================================
# 9. REMOVE INVALID HOSPITAL DATES
# ============================================================

hospital_clean = hospital_clean.dropna(
    subset=[
        "Admission_Date",
        "Discharge_Date"
    ]
).copy()


print(
    "Hospital records after date cleaning:",
    len(hospital_clean)
)


# ============================================================
# 10. CREATE HOSPITAL LENGTH OF STAY
# ============================================================

hospital_clean["Length_of_Stay"] = (
    hospital_clean["Discharge_Date"]
    -
    hospital_clean["Admission_Date"]
).dt.days


negative_hospital_stay = (
    hospital_clean["Length_of_Stay"] < 0
).sum()


print(
    "Hospital negative Length_of_Stay records:",
    negative_hospital_stay
)


hospital_clean = hospital_clean[
    hospital_clean["Length_of_Stay"] >= 0
].copy()


print(
    "Hospital Length_of_Stay created successfully."
)


# ============================================================
# 11. CONVERT HEALTHCARE DATES
# ============================================================

print("\n" + "=" * 60)
print("HEALTHCARE DATE PROCESSING")
print("=" * 60)


healthcare_clean["Date of Admission"] = pd.to_datetime(
    healthcare_clean["Date of Admission"],
    errors="coerce"
)

healthcare_clean["Discharge Date"] = pd.to_datetime(
    healthcare_clean["Discharge Date"],
    errors="coerce"
)


invalid_healthcare_dates = (
    healthcare_clean["Date of Admission"].isna()
    |
    healthcare_clean["Discharge Date"].isna()
).sum()


print(
    "Healthcare records with invalid dates:",
    invalid_healthcare_dates
)


# ============================================================
# 12. REMOVE INVALID HEALTHCARE DATES
# ============================================================

healthcare_clean = healthcare_clean.dropna(
    subset=[
        "Date of Admission",
        "Discharge Date"
    ]
).copy()


print(
    "Healthcare records after date cleaning:",
    len(healthcare_clean)
)


# ============================================================
# 13. CREATE HEALTHCARE LENGTH OF STAY
# ============================================================

healthcare_clean["Length_of_Stay"] = (
    healthcare_clean["Discharge Date"]
    -
    healthcare_clean["Date of Admission"]
).dt.days


negative_healthcare_stay = (
    healthcare_clean["Length_of_Stay"] < 0
).sum()


print(
    "Healthcare negative Length_of_Stay records:",
    negative_healthcare_stay
)


healthcare_clean = healthcare_clean[
    healthcare_clean["Length_of_Stay"] >= 0
].copy()


print(
    "Healthcare Length_of_Stay created successfully."
)


# ============================================================
# 14. CREATE BMI FOR OBESITY DATASET
# ============================================================

print("\n" + "=" * 60)
print("OBESITY FEATURE PROCESSING")
print("=" * 60)


if (
    "Height" in obesity_clean.columns
    and
    "Weight" in obesity_clean.columns
):

    obesity_clean["BMI"] = (
        obesity_clean["Weight"]
        /
        (obesity_clean["Height"] ** 2)
    )

    print(
        "BMI feature created successfully."
    )


# ============================================================
# 15. CHECK INVALID BMI VALUES
# ============================================================

if "BMI" in obesity_clean.columns:

    invalid_bmi = (
        (obesity_clean["BMI"] <= 0)
        |
        obesity_clean["BMI"].isna()
        |
        np.isinf(obesity_clean["BMI"])
    ).sum()


    print(
        "Invalid BMI records:",
        invalid_bmi
    )


    obesity_clean = obesity_clean[
        (obesity_clean["BMI"] > 0)
        &
        np.isfinite(obesity_clean["BMI"])
    ].copy()


# ============================================================
# 16. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATA QUALITY CHECK")
print("=" * 60)


print(
    "Obesity missing values:",
    obesity_clean.isnull().sum().sum()
)

print(
    "Hospital missing values:",
    hospital_clean.isnull().sum().sum()
)

print(
    "Healthcare missing values:",
    healthcare_clean.isnull().sum().sum()
)


# ============================================================
# 17. FINAL DUPLICATE CHECK
# ============================================================

print("\nFinal duplicate counts:")

print(
    "Obesity:",
    obesity_clean.duplicated().sum()
)

print(
    "Hospital:",
    hospital_clean.duplicated().sum()
)

print(
    "Healthcare:",
    healthcare_clean.duplicated().sum()
)


# ============================================================
# 18. SAVE CLEANED DATASETS
# ============================================================

obesity_output = os.path.join(
    OUTPUT_DIR,
    "obesity_cleaned.csv"
)

hospital_output = os.path.join(
    OUTPUT_DIR,
    "hospital_cleaned.csv"
)

healthcare_output = os.path.join(
    OUTPUT_DIR,
    "healthcare_cleaned.csv"
)


obesity_clean.to_csv(
    obesity_output,
    index=False
)

hospital_clean.to_csv(
    hospital_output,
    index=False
)

healthcare_clean.to_csv(
    healthcare_output,
    index=False
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)


print("\nFinal cleaned dataset shapes:")

print(
    "Obesity:",
    obesity_clean.shape
)

print(
    "Hospital:",
    hospital_clean.shape
)

print(
    "Healthcare:",
    healthcare_clean.shape
)


print("\nCleaned files saved to:")

print(obesity_output)
print(hospital_output)
print(healthcare_output)


print("\nAll preprocessing steps completed successfully!")