# ============================================================
# FEATURE SELECTION
# Healthcare BI Dashboard
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


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

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD FEATURE-ENGINEERED DATASETS
# ============================================================

obesity_path = os.path.join(
    RESULTS_DIR,
    "obesity_feature_engineered.csv"
)

hospital_path = os.path.join(
    RESULTS_DIR,
    "hospital_feature_engineered.csv"
)

healthcare_path = os.path.join(
    RESULTS_DIR,
    "healthcare_feature_engineered.csv"
)


obesity_df = pd.read_csv(obesity_path)
hospital_df = pd.read_csv(hospital_path)
healthcare_df = pd.read_csv(healthcare_path)


print("=" * 60)
print("FEATURE-ENGINEERED DATASETS LOADED")
print("=" * 60)

print("Obesity:", obesity_df.shape)
print("Hospital:", hospital_df.shape)
print("Healthcare:", healthcare_df.shape)


# ============================================================
# 3. FEATURE SELECTION FUNCTION
# ============================================================

def perform_feature_selection(
    df,
    target_column,
    dataset_name,
    columns_to_drop=None,
    top_n=10
):

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} FEATURE SELECTION")
    print("=" * 60)

    data = df.copy()

    # --------------------------------------------------------
    # Separate target
    # --------------------------------------------------------

    if target_column not in data.columns:

        print(
            f"ERROR: Target column '{target_column}' "
            f"not found."
        )

        return None

    y = data[target_column]

    X = data.drop(
        columns=[target_column]
    )


    # --------------------------------------------------------
    # Drop unnecessary columns
    # --------------------------------------------------------

    if columns_to_drop is not None:

        existing_columns = [
            column
            for column in columns_to_drop
            if column in X.columns
        ]

        X = X.drop(
            columns=existing_columns
        )


    # --------------------------------------------------------
    # Convert dates to useful numeric components
    # --------------------------------------------------------

    date_columns = X.select_dtypes(
        include=["datetime64[ns]"]
    ).columns.tolist()

    for column in date_columns:

        X[column + "_Year"] = X[column].dt.year
        X[column + "_Month"] = X[column].dt.month
        X[column + "_Day"] = X[column].dt.day

    X = X.drop(
        columns=date_columns,
        errors="ignore"
    )


    # --------------------------------------------------------
    # Convert categorical variables into dummy variables
    # --------------------------------------------------------

    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )


    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    X_encoded = X_encoded.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_encoded = X_encoded.fillna(0)


    # Make sure all columns are numeric

    X_encoded = X_encoded.astype(float)


    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y.astype(str)
    )


    # --------------------------------------------------------
    # Random Forest Feature Importance
    # --------------------------------------------------------

    print("\nTraining Random Forest for feature importance...")

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_encoded,
        y_encoded
    )


    # --------------------------------------------------------
    # Calculate feature importance
    # --------------------------------------------------------

    feature_importance = pd.DataFrame({

        "Feature": X_encoded.columns,

        "Importance": rf.feature_importances_

    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )


    # --------------------------------------------------------
    # Select top features
    # --------------------------------------------------------

    number_of_features = min(
        top_n,
        len(feature_importance)
    )

    selected_features = (
        feature_importance
        .head(number_of_features)
        ["Feature"]
        .tolist()
    )


    # --------------------------------------------------------
    # Display selected features
    # --------------------------------------------------------

    print("\nTop selected features:")

    for index, feature in enumerate(
        selected_features,
        start=1
    ):

        importance_value = feature_importance.loc[
            feature_importance["Feature"] == feature,
            "Importance"
        ].iloc[0]

        print(
            f"{index}. {feature} "
            f"-> {importance_value:.4f}"
        )


    # --------------------------------------------------------
    # Save feature importance
    # --------------------------------------------------------

    importance_output = os.path.join(
        RESULTS_DIR,
        f"{dataset_name.lower()}_feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_output,
        index=False
    )


    # --------------------------------------------------------
    # Create selected dataset
    # --------------------------------------------------------

    selected_df = X_encoded[
        selected_features
    ].copy()

    selected_df[target_column] = y.values


    # --------------------------------------------------------
    # Save selected dataset
    # --------------------------------------------------------

    selected_output = os.path.join(
        RESULTS_DIR,
        f"{dataset_name.lower()}_selected_features.csv"
    )

    selected_df.to_csv(
        selected_output,
        index=False
    )


    # --------------------------------------------------------
    # Save selected feature names
    # --------------------------------------------------------

    selected_features_output = os.path.join(
        RESULTS_DIR,
        f"{dataset_name.lower()}_selected_feature_names.csv"
    )

    pd.DataFrame({
        "Selected_Feature": selected_features
    }).to_csv(
        selected_features_output,
        index=False
    )


    print("\nFiles saved:")

    print(importance_output)

    print(selected_output)

    print(selected_features_output)


    return selected_df


# ============================================================
# 4. OBESITY FEATURE SELECTION
# ============================================================

obesity_selected = perform_feature_selection(

    df=obesity_df,

    target_column="NObeyesdad",

    dataset_name="obesity",

    columns_to_drop=[],

    top_n=10
)


# ============================================================
# 5. HOSPITAL FEATURE SELECTION
# ============================================================

hospital_selected = perform_feature_selection(

    df=hospital_df,

    target_column="Outcome",

    dataset_name="hospital",

    columns_to_drop=[
        "Patient_ID",
        "Admission_Date",
        "Discharge_Date"
    ],

    top_n=10
)


# ============================================================
# 6. HEALTHCARE FEATURE SELECTION
# ============================================================

healthcare_selected = perform_feature_selection(

    df=healthcare_df,

    target_column="Test Results",

    dataset_name="healthcare",

    columns_to_drop=[
        "Name",
        "Doctor",
        "Hospital",
        "Date of Admission",
        "Discharge Date"
    ],

    top_n=10
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETED SUCCESSFULLY")
print("=" * 60)


if obesity_selected is not None:

    print(
        "Obesity selected dataset:",
        obesity_selected.shape
    )


if hospital_selected is not None:

    print(
        "Hospital selected dataset:",
        hospital_selected.shape
    )


if healthcare_selected is not None:

    print(
        "Healthcare selected dataset:",
        healthcare_selected.shape
    )


print("\nAll feature selection steps completed!")