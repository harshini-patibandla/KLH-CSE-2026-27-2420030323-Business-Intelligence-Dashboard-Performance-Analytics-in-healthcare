# ============================================================
# HYBRID CANDIDATE 4
# Two-Stage Feature + Prediction Hybrid
# Stage 1: Random Forest
# Stage 2: XGBoost learns from original features + RF probabilities
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"D:\Healthcare-BI-Dashboard"

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "obesity_selected_features.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("HYBRID CANDIDATE 4")
print("Two-Stage RF + XGBoost Hybrid")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Shape:", df.shape)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

target = "NObeyesdad"

features = [
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

X = df[features].copy()
y = df[target].copy()


# ============================================================
# 4. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = label_encoder.classes_
num_classes = len(class_names)

print("\nTarget classes:")
print(list(class_names))


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))


# ============================================================
# 6. STAGE 1 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 — RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# RF probability outputs
rf_train_prob = rf_model.predict_proba(X_train)
rf_test_prob = rf_model.predict_proba(X_test)

print("Random Forest trained successfully.")


# ============================================================
# 7. CREATE SECOND-STAGE FEATURES
# ============================================================
# Original features + Random Forest probability outputs
#
# This is different from simple probability averaging.
# XGBoost learns how to use the RF predictions together
# with the original healthcare features.
# ============================================================

rf_prob_train_df = pd.DataFrame(
    rf_train_prob,
    columns=[f"RF_Prob_{c}" for c in class_names],
    index=X_train.index
)

rf_prob_test_df = pd.DataFrame(
    rf_test_prob,
    columns=[f"RF_Prob_{c}" for c in class_names],
    index=X_test.index
)

X_train_stage2 = pd.concat(
    [
        X_train.reset_index(drop=True),
        rf_prob_train_df.reset_index(drop=True)
    ],
    axis=1
)

X_test_stage2 = pd.concat(
    [
        X_test.reset_index(drop=True),
        rf_prob_test_df.reset_index(drop=True)
    ],
    axis=1
)

print("\nStage-2 feature shape:")
print("Train:", X_train_stage2.shape)
print("Test :", X_test_stage2.shape)


# ============================================================
# 8. STAGE 2 — XGBOOST META MODEL
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 — XGBOOST META MODEL")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.90,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_stage2,
    y_train
)

print("XGBoost meta-model trained successfully.")


# ============================================================
# 9. FINAL HYBRID PREDICTION
# ============================================================

hybrid_prob = xgb_model.predict_proba(X_test_stage2)

# Normalize probabilities to avoid numerical issues
hybrid_prob = hybrid_prob / hybrid_prob.sum(
    axis=1,
    keepdims=True
)

hybrid_pred = np.argmax(hybrid_prob, axis=1)


# ============================================================
# 10. METRICS
# ============================================================

accuracy = accuracy_score(y_test, hybrid_pred)

precision = precision_score(
    y_test,
    hybrid_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    hybrid_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    hybrid_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    hybrid_pred,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    y_test,
    hybrid_pred
)

mcc = matthews_corrcoef(
    y_test,
    hybrid_pred
)

kappa = cohen_kappa_score(
    y_test,
    hybrid_pred
)

logloss = log_loss(
    y_test,
    hybrid_prob,
    labels=np.arange(num_classes)
)

roc_auc = roc_auc_score(
    y_test,
    hybrid_prob,
    multi_class="ovr",
    average="weighted"
)


# ============================================================
# 11. SPECIFICITY
# ============================================================

cm = confusion_matrix(y_test, hybrid_pred)

specificities = []

for i in range(num_classes):

    tp = cm[i, i]

    fn = np.sum(cm[i, :]) - tp
    fp = np.sum(cm[:, i]) - tp

    tn = np.sum(cm) - (tp + fn + fp)

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0

    specificities.append(specificity)

specificity = np.mean(specificities)


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE 4 RESULTS")
print("=" * 70)

print(f"Accuracy          : {accuracy:.6f}")
print(f"Precision         : {precision:.6f}")
print(f"Recall/Sensitivity: {recall:.6f}")
print(f"F1-score          : {f1:.6f}")
print(f"Macro-F1          : {macro_f1:.6f}")
print(f"ROC-AUC           : {roc_auc:.6f}")
print(f"Specificity       : {specificity:.6f}")
print(f"Balanced Accuracy : {balanced_acc:.6f}")
print(f"MCC               : {mcc:.6f}")
print(f"Cohen Kappa       : {kappa:.6f}")
print(f"Log Loss          : {logloss:.6f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        hybrid_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

result = pd.DataFrame(
    {
        "Dataset": ["Obesity"],
        "Model": [
            "RF + XGBoost Two-Stage Hybrid"
        ],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall / Sensitivity": [recall],
        "F1-score": [f1],
        "Macro-F1": [macro_f1],
        "ROC-AUC": [roc_auc],
        "Specificity": [specificity],
        "Balanced Accuracy": [balanced_acc],
        "MCC": [mcc],
        "Cohen Kappa": [kappa],
        "Log Loss": [logloss]
    }
)

result_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate4_results.csv"
)

result.to_csv(
    result_path,
    index=False
)


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["Actual"] = label_encoder.inverse_transform(
    y_test
)

prediction_output["Predicted"] = label_encoder.inverse_transform(
    hybrid_pred
)

for i, class_name in enumerate(class_names):
    prediction_output[
        f"Probability_{class_name}"
    ] = hybrid_prob[:, i]

prediction_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate4_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 16. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate4_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("\nResults:")
print(result_path)

print("\nPredictions:")
print(prediction_path)

print("\nConfusion Matrix:")
print(cm_path)

print("\n" + "=" * 70)
print("CANDIDATE 4 COMPLETED SUCCESSFULLY")
print("=" * 70)
