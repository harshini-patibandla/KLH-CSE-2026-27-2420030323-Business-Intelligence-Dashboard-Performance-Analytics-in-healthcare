# ============================================================
# HYBRID CANDIDATE 3
# XGBoost + MLP with Validation-Based Dynamic Fusion
# Healthcare BI Dashboard - Obesity Dataset
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    classification_report
)


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
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("HYBRID CANDIDATE 3")
print("XGBoost + MLP Dynamic Probability Fusion")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("Columns:")
print(list(df.columns))


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "NObeyesdad"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 4. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = label_encoder.classes_

print("\nTarget classes:")
print(list(class_names))


# ============================================================
# 5. FEATURE GROUPS
# ============================================================
# XGBoost branch:
# Main anthropometric / obesity-related features
#
# MLP branch:
# Lifestyle / demographic features
# ============================================================

xgb_features = [
    "BMI",
    "Weight",
    "Height",
    "BMI_Category_Overweight",
    "BMI_Category_Obese",
    "BMI_Category_Underweight"
]

mlp_features = [
    "FCVC",
    "Gender_Male",
    "Age",
    "NCP"
]


# ============================================================
# 6. CHECK FEATURES
# ============================================================

missing_xgb = [
    f for f in xgb_features
    if f not in df.columns
]

missing_mlp = [
    f for f in mlp_features
    if f not in df.columns
]

if missing_xgb:
    raise ValueError(
        f"Missing XGBoost features: {missing_xgb}"
    )

if missing_mlp:
    raise ValueError(
        f"Missing MLP features: {missing_mlp}"
    )


X_xgb = df[xgb_features].copy()
X_mlp = df[mlp_features].copy()


print("\nXGBoost features:")
print(xgb_features)

print("\nMLP features:")
print(mlp_features)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_xgb_train, X_xgb_test, X_mlp_train, X_mlp_test, y_train, y_test = train_test_split(
    X_xgb,
    X_mlp,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain samples:", len(y_train))
print("Test samples :", len(y_test))


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

(
    X_xgb_train_final,
    X_xgb_val,
    X_mlp_train_final,
    X_mlp_val,
    y_train_final,
    y_val
) = train_test_split(
    X_xgb_train,
    X_mlp_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("Training samples :", len(y_train_final))
print("Validation samples:", len(y_val))


# ============================================================
# 9. XGBOOST BRANCH
# ============================================================

print("\nTraining XGBoost branch...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.90,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_xgb_train_final,
    y_train_final
)

xgb_val_prob = xgb_model.predict_proba(X_xgb_val)

print("XGBoost branch trained successfully.")


# ============================================================
# 10. MLP BRANCH
# ============================================================

print("\nTraining MLP branch...")

mlp_model = Pipeline([
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
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
    )
])

mlp_model.fit(
    X_mlp_train_final,
    y_train_final
)

mlp_val_prob = mlp_model.predict_proba(X_mlp_val)

print("MLP branch trained successfully.")


# ============================================================
# 11. VALIDATION-BASED FUSION WEIGHT SEARCH
# ============================================================

print("\nSearching for best fusion weight...")

weight_results = []

best_weight = None
best_macro_f1 = -1

for xgb_weight in np.arange(0.00, 1.01, 0.05):

    mlp_weight = 1.0 - xgb_weight

    fused_prob = (
        xgb_weight * xgb_val_prob
        +
        mlp_weight * mlp_val_prob
    )

    fused_prediction = np.argmax(
        fused_prob,
        axis=1
    )

    macro_f1 = f1_score(
        y_val,
        fused_prediction,
        average="macro"
    )

    weight_results.append({
        "XGBoost_Weight": round(xgb_weight, 2),
        "MLP_Weight": round(mlp_weight, 2),
        "Validation_Macro_F1": macro_f1
    })

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_weight = xgb_weight


best_mlp_weight = 1.0 - best_weight


print("\nBest fusion weights:")
print("XGBoost weight:", round(best_weight, 2))
print("MLP weight    :", round(best_mlp_weight, 2))

print(
    "Best validation Macro-F1:",
    round(best_macro_f1, 6)
)


# ============================================================
# 12. SAVE WEIGHT SEARCH
# ============================================================

weight_df = pd.DataFrame(weight_results)

weight_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate3_weight_search.csv"
)

weight_df.to_csv(
    weight_path,
    index=False
)


# ============================================================
# 13. WEIGHT SEARCH GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    weight_df["XGBoost_Weight"],
    weight_df["Validation_Macro_F1"],
    marker="o"
)

plt.axvline(
    best_weight,
    linestyle="--",
    label=f"Best XGBoost Weight = {best_weight:.2f}"
)

plt.xlabel("XGBoost Fusion Weight")
plt.ylabel("Validation Macro-F1")
plt.title(
    "Hybrid Candidate 3 - Fusion Weight Selection"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

weight_graph_path = os.path.join(
    GRAPHS_DIR,
    "obesity_hybrid_candidate3_weight_search.png"
)

plt.savefig(
    weight_graph_path,
    dpi=300
)

plt.close()


# ============================================================
# 14. REFIT XGBOOST ON FULL TRAINING DATA
# ============================================================

print("\nRefitting XGBoost on complete training data...")

final_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.90,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

final_xgb.fit(
    X_xgb_train,
    y_train
)


# ============================================================
# 15. REFIT MLP ON FULL TRAINING DATA
# ============================================================

print("Refitting MLP on complete training data...")

final_mlp = Pipeline([
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
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )
    )
])

final_mlp.fit(
    X_mlp_train,
    y_train
)


# ============================================================
# 16. TEST PROBABILITIES
# ============================================================

xgb_test_prob = final_xgb.predict_proba(
    X_xgb_test
)

mlp_test_prob = final_mlp.predict_proba(
    X_mlp_test
)


# ============================================================
# 17. DYNAMIC PROBABILITY FUSION
# ============================================================

final_probability = (
    best_weight * xgb_test_prob
    +
    best_mlp_weight * mlp_test_prob
)

final_prediction = np.argmax(
    final_probability,
    axis=1
)


# ============================================================
# 18. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    final_prediction
)

precision = precision_score(
    y_test,
    final_prediction,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    final_prediction,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_prediction,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    final_prediction,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    final_probability,
    multi_class="ovr",
    average="weighted"
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    final_prediction
)

mcc = matthews_corrcoef(
    y_test,
    final_prediction
)

kappa = cohen_kappa_score(
    y_test,
    final_prediction
)

logloss = log_loss(
    y_test,
    final_probability
)


# ============================================================
# 19. SPECIFICITY
# ============================================================

cm = confusion_matrix(
    y_test,
    final_prediction
)

specificities = []

for i in range(len(cm)):

    true_negative = cm.sum() - (
        cm[i, :].sum()
        +
        cm[:, i].sum()
        -
        cm[i, i]
    )

    false_positive = cm[:, i].sum() - cm[i, i]

    specificity = (
        true_negative /
        (true_negative + false_positive)
        if (true_negative + false_positive) > 0
        else 0
    )

    specificities.append(specificity)

specificity = np.mean(
    specificities
)


# ============================================================
# 20. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("HYBRID CANDIDATE 3 TEST RESULTS")
print("=" * 70)

print(f"Accuracy            : {accuracy:.6f}")
print(f"Precision           : {precision:.6f}")
print(f"Recall/Sensitivity  : {recall:.6f}")
print(f"F1 Score            : {f1:.6f}")
print(f"Macro-F1            : {macro_f1:.6f}")
print(f"ROC-AUC             : {roc_auc:.6f}")
print(f"Specificity         : {specificity:.6f}")
print(f"Balanced Accuracy   : {balanced_accuracy:.6f}")
print(f"MCC                 : {mcc:.6f}")
print(f"Cohen Kappa         : {kappa:.6f}")
print(f"Log Loss            : {logloss:.6f}")


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        final_prediction,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 22. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "Dataset": "Obesity",
    "Model": "XGBoost + MLP Dynamic Fusion",
    "XGBoost_Weight": best_weight,
    "MLP_Weight": best_mlp_weight,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall/Sensitivity": recall,
    "F1-score": f1,
    "Macro-F1": macro_f1,
    "ROC-AUC": roc_auc,
    "Specificity": specificity,
    "Balanced Accuracy": balanced_accuracy,
    "MCC": mcc,
    "Cohen Kappa": kappa,
    "Log Loss": logloss
}])


results_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate3_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 23. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(final_prediction)
})

prediction_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate3_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(9, 7))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Hybrid Candidate 3 - Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

cm_path = os.path.join(
    GRAPHS_DIR,
    "obesity_hybrid_candidate3_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("SUCCESS - HYBRID CANDIDATE 3 COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(results_path)
print(prediction_path)
print(weight_path)
print(weight_graph_path)
print(cm_path)

print("\nDO NOT call this the final proposed model yet.")
print("We will compare Candidate 3 with the existing models first.")