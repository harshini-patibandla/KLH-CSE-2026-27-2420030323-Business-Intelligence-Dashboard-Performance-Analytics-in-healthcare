# ============================================================
# MACHINE LEARNING MODELS
# Healthcare BI Dashboard
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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


# ============================================================
# 2. MODEL FUNCTION
# ============================================================

def train_models(
    file_name,
    target_column,
    dataset_name
):

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} MACHINE LEARNING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load selected features
    # --------------------------------------------------------

    file_path = os.path.join(
        RESULTS_DIR,
        file_name
    )

    df = pd.read_csv(file_path)

    print("\nDataset shape:", df.shape)

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print("Target:", target_column)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y.astype(str)
    )

    # --------------------------------------------------------
    # Make sure features are numeric
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X = X.astype(float)

    # --------------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # ========================================================
    # DECISION TREE
    # ========================================================

    print("\nTraining Decision Tree...")

    decision_tree = DecisionTreeClassifier(
        random_state=42
    )

    decision_tree.fit(
        X_train,
        y_train
    )

    dt_predictions = decision_tree.predict(
        X_test
    )

    dt_accuracy = accuracy_score(
        y_test,
        dt_predictions
    )

    dt_precision = precision_score(
        y_test,
        dt_predictions,
        average="weighted",
        zero_division=0
    )

    dt_recall = recall_score(
        y_test,
        dt_predictions,
        average="weighted",
        zero_division=0
    )

    dt_f1 = f1_score(
        y_test,
        dt_predictions,
        average="weighted",
        zero_division=0
    )

    print("Decision Tree completed.")


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print("\nTraining Random Forest...")

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_predictions = random_forest.predict(
        X_test
    )

    rf_accuracy = accuracy_score(
        y_test,
        rf_predictions
    )

    rf_precision = precision_score(
        y_test,
        rf_predictions,
        average="weighted",
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_predictions,
        average="weighted",
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_predictions,
        average="weighted",
        zero_division=0
    )

    print("Random Forest completed.")


    # ========================================================
    # MODEL RESULTS
    # ========================================================

    results = pd.DataFrame({

        "Dataset": [
            dataset_name,
            dataset_name
        ],

        "Model": [
            "Decision Tree",
            "Random Forest"
        ],

        "Accuracy": [
            dt_accuracy,
            rf_accuracy
        ],

        "Precision": [
            dt_precision,
            rf_precision
        ],

        "Recall": [
            dt_recall,
            rf_recall
        ],

        "F1_Score": [
            dt_f1,
            rf_f1
        ]
    })


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE")
    print("-" * 60)

    print(
        results.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name.lower()}_model_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)

    return results


# ============================================================
# 3. OBESITY MODEL
# ============================================================

obesity_results = train_models(

    file_name="obesity_selected_features.csv",

    target_column="NObeyesdad",

    dataset_name="Obesity"
)


# ============================================================
# 4. HOSPITAL MODEL
# ============================================================

hospital_results = train_models(

    file_name="hospital_selected_features.csv",

    target_column="Outcome",

    dataset_name="Hospital"
)


# ============================================================
# 5. HEALTHCARE MODEL
# ============================================================

healthcare_results = train_models(

    file_name="healthcare_selected_features.csv",

    target_column="Test Results",

    dataset_name="Healthcare"
)


# ============================================================
# 6. COMBINE ALL RESULTS
# ============================================================

all_results = pd.concat(
    [
        obesity_results,
        hospital_results,
        healthcare_results
    ],
    ignore_index=True
)


combined_output = os.path.join(
    RESULTS_DIR,
    "all_model_results.csv"
)

all_results.to_csv(
    combined_output,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("ALL MACHINE LEARNING MODELS COMPLETED")
print("=" * 60)

print("\nCombined model results:")

print(
    all_results.to_string(
        index=False
    )
)

print("\nCombined results saved to:")
print(combined_output)

print("\nMachine Learning phase completed successfully!")