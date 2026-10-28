# ============================================================
# HYBRID CANDIDATE 5
# Feature-Interaction Guided XGBoost + MLP
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

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

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("HYBRID CANDIDATE 5")
print("Feature-Interaction Guided XGBoost + MLP")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

target = "NObeyesdad"

base_features = [
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

X = df[base_features].copy()
y = df[target].copy()


# ============================================================
# 4. FEATURE INTERACTION ENGINEERING
# ============================================================

print("\nCreating healthcare feature interactions...")

# BMI × Age
X["BMI_Age"] = X["BMI"] * X["Age"]

# BMI × Weight
X["BMI_Weight"] = X["BMI"] * X["Weight"]

# BMI × Height
X["BMI_Height"] = X["BMI"] * X["Height"]

# Weight × Age
X["Weight_Age"] = X["Weight"] * X["Age"]

# Physical activity / food interaction
X["FCVC_NCP"] = X["FCVC"] * X["NCP"]

# BMI category interaction
X["BMI_Obesity_Signal"] = (
    X["BMI"]
    * (
        X["BMI_Category_Obese"]
        + X["BMI_Category_Overweight"]
    )
)

interaction_features = [
    "BMI_Age",
    "BMI_Weight",
    "BMI_Height",
    "Weight_Age",
    "FCVC_NCP",
    "BMI_Obesity_Signal"
]

print("Added interaction features:")
print(interaction_features)

print("\nFinal feature count:", X.shape[1])


# ============================================================
# 5. ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

class_names = encoder.classes_

num_classes = len(class_names)

print("\nTarget classes:")
print(list(class_names))


# ============================================================
# 6. TRAIN / TEST SPLIT
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
# 7. XGBOOST BRANCH
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST BRANCH")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=350,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.90,
    colsample_bytree=0.90,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_train_prob = xgb_model.predict_proba(X_train)
xgb_test_prob = xgb_model.predict_proba(X_test)

print("XGBoost branch trained successfully.")


# ============================================================
# 8. MLP BRANCH
# ============================================================

print("\n" + "=" * 70)
print("MLP BRANCH")
print("=" * 70)

mlp_model = Pipeline(
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
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.20,
                n_iter_no_change=20,
                random_state=42
            )
        )
    ]
)

mlp_model.fit(
    X_train,
    y_train
)

mlp_train_prob = mlp_model.predict_proba(X_train)
mlp_test_prob = mlp_model.predict_proba(X_test)

print("MLP branch trained successfully.")


# ============================================================
# 9. DISAGREEMENT SCORE
# ============================================================

print("\nCalculating model disagreement...")

# Absolute probability difference
prob_difference = np.abs(
    xgb_test_prob - mlp_test_prob
)

# Mean disagreement across classes
disagreement = np.mean(
    prob_difference,
    axis=1
)

print(
    "Mean disagreement:",
    np.mean(disagreement)
)

print(
    "Maximum disagreement:",
    np.max(disagreement)
)


# ============================================================
# 10. DYNAMIC FUSION
# ============================================================
# When XGBoost and MLP agree strongly:
#     trust XGBoost more.
#
# When they disagree:
#     give MLP a slightly higher contribution.
#
# The weight is determined from disagreement.
# ============================================================

min_disagreement = np.min(disagreement)
max_disagreement = np.max(disagreement)

if max_disagreement > min_disagreement:

    disagreement_norm = (
        (disagreement - min_disagreement)
        /
        (max_disagreement - min_disagreement)
    )

else:

    disagreement_norm = np.zeros_like(disagreement)


# XGBoost weight decreases slightly when disagreement increases
xgb_weight = 0.70 - (0.20 * disagreement_norm)

# MLP receives the remaining weight
mlp_weight = 1.0 - xgb_weight


# ============================================================
# 11. FINAL HYBRID PROBABILITY
# ============================================================

hybrid_prob = (
    xgb_weight[:, None] * xgb_test_prob
    +
    mlp_weight[:, None] * mlp_test_prob
)

# Normalize probabilities
hybrid_prob = (
    hybrid_prob
    /
    hybrid_prob.sum(
        axis=1,
        keepdims=True
    )
)

hybrid_pred = np.argmax(
    hybrid_prob,
    axis=1
)


# ============================================================
# 12. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    hybrid_pred
)

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
# 13. SPECIFICITY
# ============================================================

cm = confusion_matrix(
    y_test,
    hybrid_pred
)

specificities = []

for i in range(num_classes):

    tp = cm[i, i]

    fn = np.sum(cm[i, :]) - tp

    fp = np.sum(cm[:, i]) - tp

    tn = np.sum(cm) - (
        tp + fn + fp
    )

    if (tn + fp) > 0:

        spec = tn / (tn + fp)

    else:

        spec = 0

    specificities.append(spec)


specificity = np.mean(
    specificities
)


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE 5 RESULTS")
print("=" * 70)

print(
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Precision         : {precision:.6f}"
)

print(
    f"Recall/Sensitivity: {recall:.6f}"
)

print(
    f"F1-score          : {f1:.6f}"
)

print(
    f"Macro-F1          : {macro_f1:.6f}"
)

print(
    f"ROC-AUC           : {roc_auc:.6f}"
)

print(
    f"Specificity       : {specificity:.6f}"
)

print(
    f"Balanced Accuracy : {balanced_acc:.6f}"
)

print(
    f"MCC               : {mcc:.6f}"
)

print(
    f"Cohen Kappa       : {kappa:.6f}"
)

print(
    f"Log Loss          : {logloss:.6f}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
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
# 16. SAVE RESULTS
# ============================================================

result = pd.DataFrame(
    {
        "Dataset": ["Obesity"],
        "Model": [
            "Feature-Interaction XGBoost + MLP Hybrid"
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
    "obesity_hybrid_candidate5_results.csv"
)

result.to_csv(
    result_path,
    index=False
)


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["Actual"] = (
    encoder.inverse_transform(y_test)
)

prediction_output["Predicted"] = (
    encoder.inverse_transform(hybrid_pred)
)

for i, class_name in enumerate(class_names):

    prediction_output[
        f"Probability_{class_name}"
    ] = hybrid_prob[:, i]


prediction_output["Disagreement"] = disagreement

prediction_output["XGB_Weight"] = xgb_weight

prediction_output["MLP_Weight"] = mlp_weight


prediction_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate5_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 18. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate5_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# 19. SAVE DISAGREEMENT ANALYSIS
# ============================================================

disagreement_df = pd.DataFrame(
    {
        "Disagreement": disagreement,
        "XGB_Weight": xgb_weight,
        "MLP_Weight": mlp_weight,
        "Actual": encoder.inverse_transform(y_test),
        "Predicted": encoder.inverse_transform(hybrid_pred)
    }
)

disagreement_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate5_disagreement.csv"
)

disagreement_df.to_csv(
    disagreement_path,
    index=False
)


# ============================================================
# 20. FINAL MESSAGE
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

print("\nDisagreement Analysis:")
print(disagreement_path)

print("\n" + "=" * 70)
print("CANDIDATE 5 COMPLETED SUCCESSFULLY")
print("=" * 70)