
# ============================================================
# HEALTHCARE HYBRID MODEL 1
# Random Forest + XGBoost Probability Fusion
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

input_path = "outputs/results/healthcare_selected_features.csv"
results_dir = "outputs/results"

os.makedirs(results_dir, exist_ok=True)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(input_path)

target_column = "Test Results"

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' was not found.")

df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column])
y = df[target_column].astype(str)

# Remove any unexpected non-numeric feature columns
X = pd.get_dummies(X, drop_first=False)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)

# Encode target classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Dataset shape:", df.shape)
print("Features:", X.shape[1])
print("Target classes:", list(label_encoder.classes_))
print("\nClass distribution:")
print(y.value_counts())

# ------------------------------------------------------------
# 3. Split data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ------------------------------------------------------------
# 4. Train Random Forest
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. Train XGBoost
# ------------------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 6. Fuse predicted probabilities
# ------------------------------------------------------------

rf_probabilities = rf_model.predict_proba(X_test)
xgb_probabilities = xgb_model.predict_proba(X_test)

# Equal weighting: 50% Random Forest + 50% XGBoost
hybrid_probabilities = (
    0.50 * rf_probabilities +
    0.50 * xgb_probabilities
)

hybrid_predictions = np.argmax(hybrid_probabilities, axis=1)

# ------------------------------------------------------------
# 7. Evaluate hybrid model
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, hybrid_predictions)

precision = precision_score(
    y_test, hybrid_predictions,
    average="weighted", zero_division=0
)

recall = recall_score(
    y_test, hybrid_predictions,
    average="weighted", zero_division=0
)

f1 = f1_score(
    y_test, hybrid_predictions,
    average="weighted", zero_division=0
)

print("\n" + "=" * 60)
print("HEALTHCARE HYBRID MODEL 1 RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        hybrid_predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 8. Save metrics
# ------------------------------------------------------------

metrics_df = pd.DataFrame([{
    "Dataset": "Healthcare",
    "Model": "RF + XGBoost Probability Fusion",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}])

metrics_path = os.path.join(
    results_dir, "healthcare_hybrid_model_1_results.csv"
)

metrics_df.to_csv(metrics_path, index=False)

# ------------------------------------------------------------
# 9. Save confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    hybrid_predictions,
    labels=np.arange(len(label_encoder.classes_))
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_path = os.path.join(
    results_dir, "healthcare_hybrid_model_1_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

# ------------------------------------------------------------
# 10. Save predictions
# ------------------------------------------------------------

predictions_df = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(hybrid_predictions)
})

predictions_path = os.path.join(
    results_dir, "healthcare_hybrid_model_1_predictions.csv"
)

predictions_df.to_csv(predictions_path, index=False)

print("\nFiles saved successfully:")
print(metrics_path)
print(cm_path)
print(predictions_path)
