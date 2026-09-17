# ============================================================
# EXPLAINABLE AI (XAI) USING SHAP
# Healthcare BI Dashboard Project
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. FUNCTION FOR SHAP EXPLAINABILITY
# ============================================================

def run_shap_analysis(file_name, dataset_name, target_column):

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS:", dataset_name)
    print("=" * 60)

    # --------------------------------------------------------
    # Load selected features
    # --------------------------------------------------------

    file_path = os.path.join(RESULTS_DIR, file_name)

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --------------------------------------------------------
    # Convert categorical features into numbers if required
    # --------------------------------------------------------

    X = pd.get_dummies(X, drop_first=True)

    # Make sure all values are numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.fillna(0)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))

    print("Number of features:", X.shape[1])
    print("Number of classes:", len(encoder.classes_))
    print("Classes:", list(encoder.classes_))

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y_encoded)

    print("Random Forest trained successfully!")

    # --------------------------------------------------------
    # Select sample for SHAP
    # --------------------------------------------------------

    sample_size = min(500, len(X))

    X_sample = X.sample(
        n=sample_size,
        random_state=42
    )

    print("SHAP sample size:", X_sample.shape)

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(X_sample)

    # --------------------------------------------------------
    # Handle different SHAP output formats
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        shap_array = np.stack(shap_values, axis=-1)

        mean_abs_shap = np.mean(
            np.abs(shap_array),
            axis=(0, 2)
        )

    elif isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:

            mean_abs_shap = np.mean(
                np.abs(shap_values),
                axis=(0, 2)
            )

        else:

            mean_abs_shap = np.mean(
                np.abs(shap_values),
                axis=0
            )

    else:

        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:

            mean_abs_shap = np.mean(
                np.abs(shap_array),
                axis=(0, 2)
            )

        else:

            mean_abs_shap = np.mean(
                np.abs(shap_array),
                axis=0
            )

    # --------------------------------------------------------
    # Create feature importance dataframe
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP_Value": mean_abs_shap
    })

    importance_df = importance_df.sort_values(
        by="Mean_Absolute_SHAP_Value",
        ascending=False
    )

    # --------------------------------------------------------
    # Save SHAP feature importance
    # --------------------------------------------------------

    output_csv = os.path.join(
        RESULTS_DIR,
        dataset_name.lower() + "_shap_feature_importance.csv"
    )

    importance_df.to_csv(
        output_csv,
        index=False
    )

    print("SHAP feature importance saved:")
    print(output_csv)

    # --------------------------------------------------------
    # Display top features
    # --------------------------------------------------------

    print("\nTop 10 SHAP Features:")

    print(
        importance_df.head(10).to_string(index=False)
    )

    # ========================================================
    # 3. SHAP FEATURE IMPORTANCE GRAPH
    # ========================================================

    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Mean_Absolute_SHAP_Value"][::-1]
    )

    plt.title(
        f"Top 10 SHAP Feature Importance - {dataset_name}"
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Features"
    )

    plt.tight_layout()

    graph_path = os.path.join(
        GRAPHS_DIR,
        dataset_name.lower() + "_shap_feature_importance.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP graph saved:")
    print(graph_path)

    print("\nSHAP analysis completed successfully!")


# ============================================================
# 4. DATASET 1 — OBESITY
# ============================================================

run_shap_analysis(
    "obesity_selected_features.csv",
    "Obesity",
    "NObeyesdad"
)


# ============================================================
# 5. DATASET 2 — HOSPITAL
# ============================================================

run_shap_analysis(
    "hospital_selected_features.csv",
    "Hospital",
    "Outcome"
)


# ============================================================
# 6. DATASET 3 — HEALTHCARE
# ============================================================

run_shap_analysis(
    "healthcare_selected_features.csv",
    "Healthcare",
    "Test Results"
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ALL SHAP ANALYSES COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGraphs saved in:")
print(GRAPHS_DIR)

print("\nResults saved in:")
print(RESULTS_DIR)