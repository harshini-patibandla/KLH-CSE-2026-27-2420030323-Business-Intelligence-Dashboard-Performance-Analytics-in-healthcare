# ============================================================
# HYBRID CANDIDATE 2
# Feature-Partitioned RF + MLP Adaptive Fusion
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\Healthcare-BI-Dashboard"

DATA_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "obesity_selected_features.csv"
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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# ============================================================
# 2. START
# ============================================================

print("=" * 70)
print("HYBRID CANDIDATE 2")
print("Feature-Partitioned RF + MLP Adaptive Fusion")
print("=" * 70)


# ============================================================
# 3. LOAD DATA
# ============================================================

if not os.path.exists(DATA_FILE):
    print("\nERROR: Dataset file not found:")
    print(DATA_FILE)
    raise FileNotFoundError(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully")
print("File:", DATA_FILE)
print("Shape:", df.shape)

print("\nAvailable columns:")
print(list(df.columns))


# ============================================================
# 4. TARGET
# ============================================================

target = "NObeyesdad"

if target not in df.columns:

    print("\nERROR: Target column not found:")
    print(target)

    print("\nAvailable columns:")
    print(list(df.columns))

    raise ValueError(
        "NObeyesdad target column is missing."
    )


# ============================================================
# 5. REQUIRED FEATURES
# ============================================================

required_features = [
    "BMI",
    "Weight",
    "BMI_Category_Overweight",
    "BMI_Category_Obese",
    "BMI_Category_Underweight",
    "FCVC",
    "Gender_Male",
    "Age",
    "Height",
    "NCP"
]


# ============================================================
# 6. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing_features:

    print("\nERROR: These required features are missing:")
    print(missing_features)

    print("\nAvailable columns are:")
    for column in df.columns:
        print(" -", column)

    print(
        "\nThe selected-features file does not contain "
        "all the features required for Candidate 2."
    )

    raise ValueError(
        "Required features are missing from obesity_selected_features.csv."
    )


# ============================================================
# 7. SELECT FEATURES
# ============================================================

X = df[required_features].copy()

y = df[target].copy()

print("\nFeatures selected successfully:")
print(required_features)

print("\nTarget:")
print(target)


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

missing_count = X.isnull().sum().sum()

if missing_count > 0:

    print(
        "\nWARNING:",
        missing_count,
        "missing feature values found."
    )

    X = X.fillna(X.median(numeric_only=True))

else:

    print("\nNo missing feature values found.")


# ============================================================
# 9. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget classes:")
for i, class_name in enumerate(label_encoder.classes_):
    print(i, "=", class_name)


# ============================================================
# 10. FEATURE PARTITION
# ============================================================

# ------------------------------------------------------------
# RF BRANCH
# Body-composition / BMI-related features
# ------------------------------------------------------------

rf_features = [
    "BMI",
    "Weight",
    "BMI_Category_Overweight",
    "BMI_Category_Obese",
    "BMI_Category_Underweight"
]


# ------------------------------------------------------------
# MLP BRANCH
# Lifestyle / demographic features
# ------------------------------------------------------------

mlp_features = [
    "FCVC",
    "Gender_Male",
    "Age",
    "Height",
    "NCP"
]


print("\n" + "=" * 70)
print("FEATURE PARTITION")
print("=" * 70)

print("\nRandom Forest branch:")
for feature in rf_features:
    print(" -", feature)

print("\nMLP branch:")
for feature in mlp_features:
    print(" -", feature)


X_rf = X[rf_features].copy()

X_mlp = X[mlp_features].copy()


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

(
    X_rf_train,
    X_rf_test,
    X_mlp_train,
    X_mlp_test,
    y_train,
    y_test
) = train_test_split(
    X_rf,
    X_mlp,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print("Training samples:", len(y_train))
print("Testing samples :", len(y_test))


# ============================================================
# 12. TRAIN-VALIDATION SPLIT
# ============================================================

(
    X_rf_train_final,
    X_rf_validation,
    X_mlp_train_final,
    X_mlp_validation,
    y_train_final,
    y_validation
) = train_test_split(
    X_rf_train,
    X_mlp_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("Validation samples:", len(y_validation))


# ============================================================
# 13. RANDOM FOREST BRANCH
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST BRANCH")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(
    X_rf_train_final,
    y_train_final
)

rf_validation_probability = rf.predict_proba(
    X_rf_validation
)

print("Random Forest training completed.")


# ============================================================
# 14. MLP BRANCH
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MLP BRANCH")
print("=" * 70)

mlp = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        )
    ]
)

mlp.fit(
    X_mlp_train_final,
    y_train_final
)

mlp_validation_probability = mlp.predict_proba(
    X_mlp_validation
)

print("MLP training completed.")


# ============================================================
# 15. ADAPTIVE FUSION WEIGHT SEARCH
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR BEST FUSION WEIGHT")
print("=" * 70)

weight_results = []

best_weight = 0.5
best_validation_score = -1


# RF weight from 0.00 to 1.00
for rf_weight in np.arange(0.00, 1.01, 0.05):

    mlp_weight = 1.00 - rf_weight

    fused_validation_probability = (
        rf_weight * rf_validation_probability
        +
        mlp_weight * mlp_validation_probability
    )

    validation_prediction = np.argmax(
        fused_validation_probability,
        axis=1
    )

    validation_macro_f1 = f1_score(
        y_validation,
        validation_prediction,
        average="macro",
        zero_division=0
    )

    weight_results.append(
        {
            "RF_Weight": round(float(rf_weight), 2),
            "MLP_Weight": round(float(mlp_weight), 2),
            "Validation_Macro_F1": validation_macro_f1
        }
    )

    if validation_macro_f1 > best_validation_score:

        best_validation_score = validation_macro_f1

        best_weight = float(rf_weight)


best_mlp_weight = 1.00 - best_weight

weight_results_df = pd.DataFrame(
    weight_results
)


# ============================================================
# 16. SAVE WEIGHT SEARCH
# ============================================================

weight_file = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate2_weight_search.csv"
)

weight_results_df.to_csv(
    weight_file,
    index=False
)


print("\nBest fusion weight found:")

print(
    "RF weight :",
    round(best_weight, 2)
)

print(
    "MLP weight:",
    round(best_mlp_weight, 2)
)

print(
    "Validation Macro-F1:",
    round(best_validation_score, 6)
)


# ============================================================
# 17. WEIGHT SEARCH GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    weight_results_df["RF_Weight"],
    weight_results_df["Validation_Macro_F1"],
    marker="o"
)

plt.xlabel("Random Forest Weight")

plt.ylabel("Validation Macro-F1")

plt.title(
    "Adaptive Fusion Weight Search"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

weight_graph = os.path.join(
    GRAPHS_DIR,
    "obesity_hybrid_candidate2_weight_search.png"
)

plt.savefig(
    weight_graph,
    dpi=300
)

plt.close()


# ============================================================
# 18. REFIT RANDOM FOREST ON FULL TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("REFITTING FINAL BRANCH MODELS")
print("=" * 70)

rf_final = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_final.fit(
    X_rf_train,
    y_train
)


# ============================================================
# 19. REFIT MLP ON FULL TRAINING DATA
# ============================================================

mlp_final = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        )
    ]
)

mlp_final.fit(
    X_mlp_train,
    y_train
)

print("Final branch models trained.")


# ============================================================
# 20. TEST PROBABILITIES
# ============================================================

rf_test_probability = rf_final.predict_proba(
    X_rf_test
)

mlp_test_probability = mlp_final.predict_proba(
    X_mlp_test
)


# ============================================================
# 21. ADAPTIVE PROBABILITY FUSION
# ============================================================

hybrid_test_probability = (
    best_weight * rf_test_probability
    +
    best_mlp_weight * mlp_test_probability
)


# ============================================================
# 22. FINAL PREDICTION
# ============================================================

hybrid_prediction = np.argmax(
    hybrid_test_probability,
    axis=1
)


# ============================================================
# 23. BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    hybrid_prediction
)

precision = precision_score(
    y_test,
    hybrid_prediction,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    hybrid_prediction,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    hybrid_prediction,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    hybrid_prediction,
    average="macro",
    zero_division=0
)


# ============================================================
# 24. ADDITIONAL METRICS
# ============================================================

balanced_accuracy = balanced_accuracy_score(
    y_test,
    hybrid_prediction
)

mcc = matthews_corrcoef(
    y_test,
    hybrid_prediction
)

kappa = cohen_kappa_score(
    y_test,
    hybrid_prediction
)

logloss = log_loss(
    y_test,
    hybrid_test_probability
)

roc_auc = roc_auc_score(
    y_test,
    hybrid_test_probability,
    multi_class="ovr",
    average="weighted"
)


# ============================================================
# 25. SPECIFICITY
# ============================================================

confusion = confusion_matrix(
    y_test,
    hybrid_prediction
)

specificity_values = []

for class_index in range(
    len(label_encoder.classes_)
):

    true_negative = (
        np.sum(confusion)
        -
        np.sum(confusion[class_index, :])
        -
        np.sum(confusion[:, class_index])
        +
        confusion[class_index, class_index]
    )

    false_positive = (
        np.sum(confusion[:, class_index])
        -
        confusion[class_index, class_index]
    )

    denominator = (
        true_negative
        +
        false_positive
    )

    if denominator > 0:

        specificity = (
            true_negative / denominator
        )

    else:

        specificity = 0

    specificity_values.append(
        specificity
    )


macro_specificity = np.mean(
    specificity_values
)


# ============================================================
# 26. RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        {
            "Dataset": "Obesity",

            "Model":
            "Feature-Partitioned RF + MLP Adaptive Fusion",

            "RF_Weight":
            best_weight,

            "MLP_Weight":
            best_mlp_weight,

            "Accuracy":
            accuracy,

            "Precision":
            precision,

            "Recall":
            recall,

            "F1-score":
            f1,

            "Macro-F1":
            macro_f1,

            "ROC-AUC":
            roc_auc,

            "Specificity":
            macro_specificity,

            "Balanced Accuracy":
            balanced_accuracy,

            "MCC":
            mcc,

            "Cohen Kappa":
            kappa,

            "Log Loss":
            logloss
        }
    ]
)


# ============================================================
# 27. SAVE RESULTS
# ============================================================

result_file = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate2_results.csv"
)

results.to_csv(
    result_file,
    index=False
)


# ============================================================
# 28. SAVE PREDICTIONS
# ============================================================

prediction_output = pd.DataFrame(
    {
        "Actual":
        label_encoder.inverse_transform(
            y_test
        ),

        "Predicted":
        label_encoder.inverse_transform(
            hybrid_prediction
        )
    }
)

prediction_file = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate2_predictions.csv"
)

prediction_output.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# 29. CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    confusion
)

plt.title(
    "Feature-Partitioned RF + MLP Adaptive Fusion\n"
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)


for i in range(
    confusion.shape[0]
):

    for j in range(
        confusion.shape[1]
    ):

        plt.text(
            j,
            i,
            confusion[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

confusion_graph = os.path.join(
    GRAPHS_DIR,
    "obesity_hybrid_candidate2_confusion_matrix.png"
)

plt.savefig(
    confusion_graph,
    dpi=300
)

plt.close()


# ============================================================
# 30. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    "\nModel:"
)

print(
    "Feature-Partitioned RF + MLP Adaptive Fusion"
)

print(
    "\nAdaptive Fusion Weights:"
)

print(
    "Random Forest:",
    round(best_weight, 4)
)

print(
    "MLP:",
    round(best_mlp_weight, 4)
)


print(
    "\nPerformance Metrics:"
)

print(
    "Accuracy          :",
    round(accuracy, 6)
)

print(
    "Precision         :",
    round(precision, 6)
)

print(
    "Recall            :",
    round(recall, 6)
)

print(
    "F1-score          :",
    round(f1, 6)
)

print(
    "Macro-F1          :",
    round(macro_f1, 6)
)

print(
    "ROC-AUC           :",
    round(roc_auc, 6)
)

print(
    "Specificity       :",
    round(macro_specificity, 6)
)

print(
    "Balanced Accuracy :",
    round(balanced_accuracy, 6)
)

print(
    "MCC               :",
    round(mcc, 6)
)

print(
    "Cohen Kappa       :",
    round(kappa, 6)
)

print(
    "Log Loss          :",
    round(logloss, 6)
)


# ============================================================
# 31. CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        hybrid_prediction,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 32. OUTPUT FILES
# ============================================================

print(
    "\nFiles created:"
)

print(
    result_file
)

print(
    prediction_file
)

print(
    weight_file
)

print(
    weight_graph
)

print(
    confusion_graph
)


print(
    "\n" + "=" * 70
)

print(
    "CANDIDATE 2 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)