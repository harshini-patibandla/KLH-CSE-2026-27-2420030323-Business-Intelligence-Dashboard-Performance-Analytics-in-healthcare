
# ============================================================
# PROPOSED HYBRID MODEL
# Random Forest + MLP Probability Fusion
# Dataset: Hospital Patient
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "outputs", "results",
    "hospital_selected_features.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("HOSPITAL PATIENT HYBRID MODEL")
print("Random Forest + MLP Probability Fusion")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check the filename in outputs/results."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ============================================================
# 3. TARGET AND FEATURES
# ============================================================

TARGET = "Outcome"

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[TARGET]).copy()

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Remove rows with missing input values.
# This simple version uses median/mode imputation.
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(
            X[col].mode().iloc[0] if not X[col].mode().empty
            else "Unknown"
        )

# Encode categorical input features.
X = pd.get_dummies(X, drop_first=False)

# Encode target classes.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

print("\nTarget classes:")
for i, name in enumerate(label_encoder.classes_):
    print(i, "=", name)

# ============================================================
# 4. TRAIN-TEST SPLIT
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
# 5. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
rf_probabilities = rf_model.predict_proba(X_test)

print("Random Forest completed.")

# ============================================================
# 6. MLP
# ============================================================

print("\nTraining MLP...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True
)

mlp_model.fit(X_train_scaled, y_train)
mlp_probabilities = mlp_model.predict_proba(X_test_scaled)

print("MLP completed.")

# ============================================================
# 7. HYBRID PROBABILITY FUSION
# ============================================================

RF_WEIGHT = 0.50
MLP_WEIGHT = 0.50

hybrid_probabilities = (
    RF_WEIGHT * rf_probabilities
    + MLP_WEIGHT * mlp_probabilities
)

hybrid_predictions = np.argmax(hybrid_probabilities, axis=1)

# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

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

try:
    if len(label_encoder.classes_) == 2:
        roc_auc = roc_auc_score(
            y_test, hybrid_probabilities[:, 1]
        )
    else:
        roc_auc = roc_auc_score(
            y_test,
            hybrid_probabilities,
            multi_class="ovr",
            average="weighted"
        )
except ValueError:
    roc_auc = np.nan

print("\n" + "=" * 70)
print("HOSPITAL PATIENT HYBRID RESULTS")
print("=" * 70)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}" if not np.isnan(roc_auc)
      else "ROC-AUC   : Not available")

print("\nClassification Report:")
print(classification_report(
    y_test,
    hybrid_predictions,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "Dataset": "Hospital Patient",
    "Model": "Random Forest + MLP Hybrid",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "ROC-AUC": roc_auc
}])

results_path = os.path.join(
    OUTPUT_DIR, "hospital_hybrid_model_results.csv"
)
results.to_csv(results_path, index=False)

predictions = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(hybrid_predictions)
})

predictions_path = os.path.join(
    OUTPUT_DIR, "hospital_hybrid_predictions.csv"
)
predictions.to_csv(predictions_path, index=False)

cm = confusion_matrix(
    y_test,
    hybrid_predictions,
    labels=np.arange(len(label_encoder.classes_))
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in label_encoder.classes_],
    columns=[f"Predicted_{c}" for c in label_encoder.classes_]
)

cm_path = os.path.join(
    OUTPUT_DIR, "hospital_hybrid_confusion_matrix.csv"
)
cm_df.to_csv(cm_path)

print("\nSaved results:")
print(results_path)
print(predictions_path)
print(cm_path)

print("\nHospital Patient hybrid experiment completed.")
