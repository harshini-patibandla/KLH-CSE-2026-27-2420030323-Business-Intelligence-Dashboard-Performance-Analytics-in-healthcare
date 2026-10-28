# ============================================================
# HYBRID 6
# Domain-Guided Feature-View XGBoost + MLP + Meta-Learner
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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
    log_loss
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"D:\Healthcare-BI-Dashboard"

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned_data",
    "obesity_cleaned.csv"
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
print("HYBRID 6 — DOMAIN-GUIDED FEATURE-VIEW HYBRID")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "NObeyesdad"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# Convert categorical input columns to numeric
X = pd.get_dummies(X, drop_first=False)

# Make sure all values are numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nNumber of features after encoding:", X.shape[1])
print("Number of classes:", len(label_encoder.classes_))
print("Classes:", list(label_encoder.classes_))


# ============================================================
# 5. DOMAIN-GUIDED FEATURE GROUPS
# ============================================================

# Original/domain feature names that may exist after encoding.
# We search by name so encoded categorical columns are also handled.

anthropometric_keywords = [
    "BMI",
    "Weight",
    "Height",
    "Age"
]

lifestyle_keywords = [
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]

anthropometric_features = []
lifestyle_features = []
other_features = []

for column in X.columns:

    if any(column.startswith(k) for k in anthropometric_keywords):
        anthropometric_features.append(column)

    elif any(column.startswith(k) for k in lifestyle_keywords):
        lifestyle_features.append(column)

    else:
        other_features.append(column)


print("\n" + "=" * 70)
print("DOMAIN FEATURE GROUPS")
print("=" * 70)

print("\nAnthropometric features:")
print(anthropometric_features)

print("\nLifestyle features:")
print(lifestyle_features)

print("\nOther features:")
print(other_features)


# Safety check
if len(anthropometric_features) == 0:
    raise ValueError("No anthropometric features were found.")

if len(lifestyle_features) == 0:
    raise ValueError("No lifestyle features were found.")


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. CREATE FEATURE VIEWS
# ============================================================

X_train_anthro = X_train[anthropometric_features]
X_test_anthro = X_test[anthropometric_features]

X_train_lifestyle = X_train[lifestyle_features]
X_test_lifestyle = X_test[lifestyle_features]


# ============================================================
# 8. XGBOOST — ANTHROPOMETRIC VIEW
# ============================================================

print("\nTraining XGBoost on anthropometric features...")

num_classes = len(label_encoder.classes_)

xgb_model = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(
    X_train_anthro,
    y_train
)

xgb_train_prob = xgb_model.predict_proba(X_train_anthro)
xgb_test_prob = xgb_model.predict_proba(X_test_anthro)


# ============================================================
# 9. MLP — LIFESTYLE VIEW
# ============================================================

print("Training MLP on lifestyle features...")

scaler = StandardScaler()

X_train_lifestyle_scaled = scaler.fit_transform(
    X_train_lifestyle
)

X_test_lifestyle_scaled = scaler.transform(
    X_test_lifestyle
)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0005,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)

mlp_model.fit(
    X_train_lifestyle_scaled,
    y_train
)

mlp_train_prob = mlp_model.predict_proba(
    X_train_lifestyle_scaled
)

mlp_test_prob = mlp_model.predict_proba(
    X_test_lifestyle_scaled
)


# ============================================================
# 10. META-LEARNER
# ============================================================

print("Training Logistic Regression meta-learner...")

meta_train = np.hstack([
    xgb_train_prob,
    mlp_train_prob
])

meta_test = np.hstack([
    xgb_test_prob,
    mlp_test_prob
])

meta_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

meta_model.fit(
    meta_train,
    y_train
)

final_prob = meta_model.predict_proba(meta_test)

final_pred = np.argmax(
    final_prob,
    axis=1
)


# ============================================================
# 11. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    final_pred
)

precision = precision_score(
    y_test,
    final_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    final_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    final_pred,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    y_test,
    final_pred
)

mcc = matthews_corrcoef(
    y_test,
    final_pred
)

kappa = cohen_kappa_score(
    y_test,
    final_pred
)

logloss = log_loss(
    y_test,
    final_prob
)

roc_auc = roc_auc_score(
    y_test,
    final_prob,
    multi_class="ovr",
    average="weighted"
)


# ============================================================
# 12. SPECIFICITY
# ============================================================

cm = confusion_matrix(y_test, final_pred)

specificities = []

for i in range(len(cm)):

    true_negative = np.sum(cm) - (
        np.sum(cm[i, :]) +
        np.sum(cm[:, i]) -
        cm[i, i]
    )

    false_positive = np.sum(cm[:, i]) - cm[i, i]

    denominator = true_negative + false_positive

    if denominator > 0:
        specificity_i = true_negative / denominator
    else:
        specificity_i = 0

    specificities.append(specificity_i)

specificity = np.mean(specificities)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("HYBRID 6 RESULTS")
print("=" * 70)

print(f"Accuracy           : {accuracy:.6f}")
print(f"Precision          : {precision:.6f}")
print(f"Recall             : {recall:.6f}")
print(f"F1 Score           : {f1:.6f}")
print(f"Macro-F1           : {macro_f1:.6f}")
print(f"ROC-AUC            : {roc_auc:.6f}")
print(f"Specificity        : {specificity:.6f}")
print(f"Balanced Accuracy  : {balanced_acc:.6f}")
print(f"MCC                : {mcc:.6f}")
print(f"Cohen's Kappa      : {kappa:.6f}")
print(f"Log Loss           : {logloss:.6f}")


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "Model": "Hybrid 6 - Domain-Guided XGB + MLP + Meta-Learner",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Macro-F1": macro_f1,
    "ROC-AUC": roc_auc,
    "Specificity": specificity,
    "Balanced Accuracy": balanced_acc,
    "MCC": mcc,
    "Kappa": kappa,
    "Log Loss": logloss
}])

results_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate6_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 15. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate6_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(final_pred)
})

prediction_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate6_predictions.csv"
)

predictions.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 17. SAVE FEATURE GROUP INFORMATION
# ============================================================

feature_groups = pd.DataFrame({
    "Feature_Group": (
        ["Anthropometric"] * len(anthropometric_features)
        + ["Lifestyle"] * len(lifestyle_features)
        + ["Other"] * len(other_features)
    ),
    "Feature": (
        anthropometric_features
        + lifestyle_features
        + other_features
    )
})

feature_group_path = os.path.join(
    RESULTS_DIR,
    "obesity_hybrid_candidate6_feature_groups.csv"
)

feature_groups.to_csv(
    feature_group_path,
    index=False
)


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_path)
print(cm_path)
print(prediction_path)
print(feature_group_path)

print("\nHYBRID 6 COMPLETED SUCCESSFULLY!")
print("=" * 70)