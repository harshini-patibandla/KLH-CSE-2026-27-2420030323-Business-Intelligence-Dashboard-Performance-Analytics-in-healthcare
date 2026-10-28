# ============================================================
# MODEL EVALUATION
# Healthcare BI Dashboard
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
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
# 2. METRIC CALCULATION FUNCTION
# ============================================================

def calculate_metrics(
    model,
    X_test,
    y_test,
    predictions
):

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Recall / Sensitivity
    # --------------------------------------------------------

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # F1-score
    # --------------------------------------------------------

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        probabilities = model.predict_proba(
            X_test
        )

        number_of_classes = len(
            np.unique(y_test)
        )

        # Binary classification
        if number_of_classes == 2:

            roc_auc = roc_auc_score(
                y_test,
                probabilities[:, 1]
            )

        # Multiclass classification
        else:

            roc_auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="weighted"
            )

    except Exception as e:

        print(
            "ROC-AUC calculation warning:",
            e
        )

        roc_auc = np.nan

    return (
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(
    file_name,
    target_column,
    dataset_name
):

    print("\n" + "=" * 70)

    print(
        f"{dataset_name.upper()} MODEL EVALUATION"
    )

    print("=" * 70)


    # ========================================================
    # LOAD DATA
    # ========================================================

    file_path = os.path.join(
        RESULTS_DIR,
        file_name
    )

    df = pd.read_csv(
        file_path
    )

    print(
        "\nDataset shape:",
        df.shape
    )

    print(
        "Target:",
        target_column
    )


    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    # ========================================================
    # ENCODE TARGET
    # ========================================================

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y.astype(str)
    )


    # ========================================================
    # PREPARE FEATURES
    # ========================================================

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


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

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
    # MODEL DEFINITIONS
    # ========================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss"
            ),

        "CatBoost":
            CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                verbose=0,
                random_seed=42
            ),

        "MLP":
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=42
            )
    }


    # ========================================================
    # STANDARD SCALING FOR MLP
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    results = []


    # ========================================================
    # TRAIN AND EVALUATE ALL MODELS
    # ========================================================

    for model_name, model in models.items():

        print("\n" + "-" * 60)

        print(
            f"Training {model_name}..."
        )

        print("-" * 60)


        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        if model_name == "MLP":

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            (
                accuracy,
                precision,
                recall,
                f1,
                roc_auc
            ) = calculate_metrics(
                model,
                X_test_scaled,
                y_test,
                predictions
            )


        # ----------------------------------------------------
        # Other models
        # ----------------------------------------------------

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            (
                accuracy,
                precision,
                recall,
                f1,
                roc_auc
            ) = calculate_metrics(
                model,
                X_test,
                y_test,
                predictions
            )


        # ====================================================
        # STORE RESULTS
        # ====================================================

        results.append({

            "Dataset":
                dataset_name,

            "Model":
                model_name,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall / Sensitivity":
                recall,

            "F1-score":
                f1,

            "ROC-AUC":
                roc_auc
        })


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"Accuracy:    {accuracy:.4f}"
        )

        print(
            f"Precision:   {precision:.4f}"
        )

        print(
            f"Recall:      {recall:.4f}"
        )

        print(
            f"F1-score:    {f1:.4f}"
        )

        print(
            f"ROC-AUC:     {roc_auc:.4f}"
        )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        print(
            f"\n{model_name} Classification Report:"
        )

        print(
            classification_report(
                y_test,
                predictions,
                labels=range(
                    len(encoder.classes_)
                ),
                target_names=encoder.classes_,
                zero_division=0
            )
        )


    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    # ========================================================
    # SAVE DATASET RESULTS
    # ========================================================

    result_file = os.path.join(
        RESULTS_DIR,
        f"{dataset_name.lower()}_evaluation_metrics.csv"
    )

    results_df.to_csv(
        result_file,
        index=False
    )

    print(
        "\nSaved metrics:",
        result_file
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        "\n" + "=" * 100
    )

    print(
        f"{dataset_name} PERFORMANCE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        results_df.round(4).to_string(
            index=False
        )
    )


    # ========================================================
    # DECISION TREE CONFUSION MATRIX
    # ========================================================

    dt = models[
        "Decision Tree"
    ]

    dt_predictions = dt.predict(
        X_test
    )

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
    # RANDOM FOREST CONFUSION MATRIX
    # ========================================================

    rf = models[
        "Random Forest"
    ]

    rf_predictions = rf.predict(
        X_test
    )

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
    # RANDOM FOREST ROC CURVE
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
    # Binary Classification
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
    # Multiclass Classification
    # --------------------------------------------------------

    else:

        y_test_binary = label_binarize(
            y_test,
            classes=range(
                number_of_classes
            )
        )

        for i in range(
            number_of_classes
        ):

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
    # Random Baseline
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


    return results_df


# ============================================================
# 4. OBESITY
# ============================================================

obesity_results = evaluate_dataset(
    file_name="obesity_selected_features.csv",
    target_column="NObeyesdad",
    dataset_name="Obesity"
)


# ============================================================
# 5. HOSPITAL
# ============================================================

hospital_results = evaluate_dataset(
    file_name="hospital_selected_features.csv",
    target_column="Outcome",
    dataset_name="Hospital"
)


# ============================================================
# 6. HEALTHCARE
# ============================================================

healthcare_results = evaluate_dataset(
    file_name="healthcare_selected_features.csv",
    target_column="Test Results",
    dataset_name="Healthcare"
)


# ============================================================
# 7. COMBINE ALL RESULTS
# ============================================================

all_evaluation_results = pd.concat(
    [
        obesity_results,
        hospital_results,
        healthcare_results
    ],
    ignore_index=True
)


# ============================================================
# 8. SAVE ALL EVALUATION RESULTS
# ============================================================

all_results_path = os.path.join(
    RESULTS_DIR,
    "all_evaluation_metrics.csv"
)

all_evaluation_results.to_csv(
    all_results_path,
    index=False
)


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL EVALUATION COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    "\nAll evaluation metrics saved to:"
)

print(
    all_results_path
)

print(
    "\nMetrics included:"
)

print(
    "1. Accuracy"
)

print(
    "2. Precision"
)

print(
    "3. Recall / Sensitivity"
)

print(
    "4. F1-score"
)

print(
    "5. ROC-AUC"
)

print(
    "\nEvaluation graphs saved in:"
)

print(
    GRAPHS_DIR
)