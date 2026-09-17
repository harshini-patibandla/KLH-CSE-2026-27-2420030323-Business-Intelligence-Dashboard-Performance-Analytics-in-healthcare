# ============================================================
# MODEL EVALUATION
# Healthcare BI Dashboard
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)


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

GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "graphs"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPHS_DIR,
    exist_ok=True
)


# ============================================================
# 2. EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(
    file_name,
    target_column,
    dataset_name
):

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} MODEL EVALUATION")
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
    print("Target:", target_column)

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y.astype(str)
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    X = X.astype(float)

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    # ========================================================
    # DECISION TREE
    # ========================================================

    print("\nTraining Decision Tree...")

    dt = DecisionTreeClassifier(
        random_state=42
    )

    dt.fit(
        X_train,
        y_train
    )

    dt_predictions = dt.predict(
        X_test
    )

    print("Decision Tree completed.")

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_predictions = rf.predict(
        X_test
    )

    print("Random Forest completed.")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nDecision Tree Classification Report:"
    )

    print(
        classification_report(
            y_test,
            dt_predictions,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    print(
        "\nRandom Forest Classification Report:"
    )

    print(
        classification_report(
            y_test,
            rf_predictions,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX - DECISION TREE
    # ========================================================

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        dt_predictions,
        display_labels=encoder.classes_,
        xticks_rotation=45
    )

    plt.title(
        f"{dataset_name} - Decision Tree Confusion Matrix"
    )

    plt.tight_layout()

    dt_cm_path = os.path.join(
        GRAPHS_DIR,
        f"{dataset_name.lower()}_decision_tree_confusion_matrix.png"
    )

    plt.savefig(
        dt_cm_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        dt_cm_path
    )

    # ========================================================
    # CONFUSION MATRIX - RANDOM FOREST
    # ========================================================

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        rf_predictions,
        display_labels=encoder.classes_,
        xticks_rotation=45
    )

    plt.title(
        f"{dataset_name} - Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    rf_cm_path = os.path.join(
        GRAPHS_DIR,
        f"{dataset_name.lower()}_random_forest_confusion_matrix.png"
    )

    plt.savefig(
        rf_cm_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        rf_cm_path
    )

    # ========================================================
    # ROC CURVE
    # ========================================================

    number_of_classes = len(
        encoder.classes_
    )

    rf_probabilities = rf.predict_proba(
        X_test
    )

    plt.figure(
        figsize=(8, 6)
    )

    # --------------------------------------------------------
    # Binary classification
    # --------------------------------------------------------

    if number_of_classes == 2:

        fpr, tpr, _ = roc_curve(
            y_test,
            rf_probabilities[:, 1]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.2f}"
        )

    # --------------------------------------------------------
    # Multiclass classification
    # --------------------------------------------------------

    else:

        y_test_binary = label_binarize(
            y_test,
            classes=range(number_of_classes)
        )

        for i in range(number_of_classes):

            fpr, tpr, _ = roc_curve(
                y_test_binary[:, i],
                rf_probabilities[:, i]
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            plt.plot(
                fpr,
                tpr,
                label=(
                    f"{encoder.classes_[i]} "
                    f"(AUC = {roc_auc:.2f})"
                )
            )

    # --------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title(
        f"{dataset_name} - Random Forest ROC Curve"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    roc_path = os.path.join(
        GRAPHS_DIR,
        f"{dataset_name.lower()}_random_forest_roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        roc_path
    )


# ============================================================
# 3. OBESITY
# ============================================================

evaluate_dataset(
    file_name="obesity_selected_features.csv",
    target_column="NObeyesdad",
    dataset_name="Obesity"
)


# ============================================================
# 4. HOSPITAL
# ============================================================

evaluate_dataset(
    file_name="hospital_selected_features.csv",
    target_column="Outcome",
    dataset_name="Hospital"
)


# ============================================================
# 5. HEALTHCARE
# ============================================================

evaluate_dataset(
    file_name="healthcare_selected_features.csv",
    target_column="Test Results",
    dataset_name="Healthcare"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nEvaluation graphs saved in:"
)

print(
    GRAPHS_DIR
)