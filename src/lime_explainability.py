# ============================================================
# LIME EXPLAINABILITY
# Healthcare BI Dashboard
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# ============================================================
# FUNCTION FOR LIME ANALYSIS
# ============================================================

def perform_lime_analysis(dataset_name, file_name, target_column):

    print("\n" + "=" * 60)
    print(f"LIME ANALYSIS: {dataset_name}")
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

    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found.")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --------------------------------------------------------
    # Convert categorical features
    # --------------------------------------------------------

    X = pd.get_dummies(X)

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace missing/infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_names = list(X.columns)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Number of features:", len(feature_names))
    print("Number of classes:", len(encoder.classes_))
    print("Classes:", list(encoder.classes_))

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y_encoded)

    print("Random Forest trained successfully!")

    # --------------------------------------------------------
    # Create LIME Explainer
    # --------------------------------------------------------

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=feature_names,
        class_names=list(encoder.classes_),
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    # --------------------------------------------------------
    # Select one representative sample
    # --------------------------------------------------------

    sample_index = 0
    sample = X.iloc[sample_index].values

    actual_class = encoder.inverse_transform(
        [y_encoded[sample_index]]
    )[0]

    predicted_class_index = model.predict(
        sample.reshape(1, -1)
    )[0]

    predicted_class = encoder.inverse_transform(
        [predicted_class_index]
    )[0]

    print("\nActual class:", actual_class)
    print("Predicted class:", predicted_class)

    # --------------------------------------------------------
    # Generate LIME explanation
    # --------------------------------------------------------

    print("Generating LIME explanation...")

    explanation = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=min(10, len(feature_names)),
        top_labels=1
    )

    # --------------------------------------------------------
    # Get explanation for predicted class
    # --------------------------------------------------------

    label_to_explain = predicted_class_index

    explanation_list = explanation.as_list(
        label=label_to_explain
    )

    # --------------------------------------------------------
    # Save explanation results
    # --------------------------------------------------------

    explanation_df = pd.DataFrame(
        explanation_list,
        columns=["Feature", "LIME_Weight"]
    )

    explanation_output = os.path.join(
        RESULTS_DIR,
        f"{dataset_name.lower()}_lime_explanation.csv"
    )

    explanation_df.to_csv(
        explanation_output,
        index=False
    )

    print("\nLIME explanation saved:")
    print(explanation_output)

    # --------------------------------------------------------
    # Display explanation
    # --------------------------------------------------------

    print("\nTop LIME Features:")

    print(explanation_df)

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plot_df = explanation_df.copy()

    plot_df = plot_df.sort_values(
        by="LIME_Weight"
    )

    plt.barh(
        plot_df["Feature"],
        plot_df["LIME_Weight"]
    )

    plt.xlabel("LIME Weight")
    plt.ylabel("Feature")

    plt.title(
        f"LIME Explanation - {dataset_name}"
    )

    plt.tight_layout()

    graph_output = os.path.join(
        GRAPHS_DIR,
        f"{dataset_name.lower()}_lime_explanation.png"
    )

    plt.savefig(
        graph_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nLIME graph saved:")
    print(graph_output)

    print("\nLIME analysis completed successfully!")


# ============================================================
# DATASET 1 — OBESITY
# ============================================================

perform_lime_analysis(
    dataset_name="Obesity",
    file_name="obesity_selected_features.csv",
    target_column="NObeyesdad"
)


# ============================================================
# DATASET 2 — HOSPITAL
# ============================================================

perform_lime_analysis(
    dataset_name="Hospital",
    file_name="hospital_selected_features.csv",
    target_column="Outcome"
)


# ============================================================
# DATASET 3 — HEALTHCARE
# ============================================================

perform_lime_analysis(
    dataset_name="Healthcare",
    file_name="healthcare_selected_features.csv",
    target_column="Test Results"
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ALL LIME ANALYSES COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGraphs saved in:")
print(GRAPHS_DIR)

print("\nResults saved in:")
print(RESULTS_DIR)