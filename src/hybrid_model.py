# ============================================================
# PROPOSED HYBRID MODEL
# Random Forest + MLP Probability Fusion
# Dataset: Obesity
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
    classification_report
)

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "obesity_selected_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("PROPOSED HYBRID MODEL")
print("Random Forest + MLP Probability Fusion")
print("=" * 80)

print("\nLoading Obesity selected-features dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. TARGET COLUMN
# ============================================================

TARGET = "NObeyesdad"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\nEncoding categorical features...")

X = pd.get_dummies(X, drop_first=False)

print("Encoded feature shape:", X.shape)


# ============================================================
# 6. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget classes:")
for i, class_name in enumerate(label_encoder.classes_):
    print(i, "=", class_name)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 8. RANDOM FOREST BRANCH
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST BRANCH")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_probabilities = rf_model.predict_proba(X_test)

print("Random Forest training completed.")


# ============================================================
# 9. MLP BRANCH
# ============================================================

print("\n" + "=" * 80)
print("TRAINING MLP DEEP LEARNING BRANCH")
print("=" * 80)

# MLP requires scaled input
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

mlp_model.fit(X_train_scaled, y_train)

mlp_probabilities = mlp_model.predict_proba(X_test_scaled)

print("MLP training completed.")


# ============================================================
# 10. HYBRID PROBABILITY FUSION
# ============================================================

print("\n" + "=" * 80)
print("HYBRID PROBABILITY FUSION")
print("=" * 80)

# Equal-weight probability fusion
RF_WEIGHT = 0.50
MLP_WEIGHT = 0.50

hybrid_probabilities = (
    RF_WEIGHT * rf_probabilities
    +
    MLP_WEIGHT * mlp_probabilities
)

# Select class having highest fused probability
hybrid_predictions = np.argmax(
    hybrid_probabilities,
    axis=1
)

print("Random Forest weight:", RF_WEIGHT)
print("MLP weight:", MLP_WEIGHT)

print("Hybrid prediction completed.")


# ============================================================
# 11. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    hybrid_predictions
)

precision = precision_score(
    y_test,
    hybrid_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    hybrid_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    hybrid_predictions,
    average="weighted",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    hybrid_probabilities,
    multi_class="ovr",
    average="weighted"
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("HYBRID MODEL RESULTS")
print("=" * 80)

print(f"Accuracy              : {accuracy:.4f}")
print(f"Precision             : {precision:.4f}")
print(f"Recall / Sensitivity  : {recall:.4f}")
print(f"F1-score              : {f1:.4f}")
print(f"ROC-AUC               : {roc_auc:.4f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        hybrid_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 14. SAVE HYBRID MODEL RESULTS
# ============================================================

hybrid_results = pd.DataFrame({
    "Dataset": ["Obesity"],
    "Model": ["Random Forest + MLP Hybrid"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall / Sensitivity": [recall],
    "F1-score": [f1],
    "ROC-AUC": [roc_auc]
})

output_path = os.path.join(
    OUTPUT_DIR,
    "obesity_hybrid_model_results.csv"
)

hybrid_results.to_csv(
    output_path,
    index=False
)

print("\nHybrid model results saved to:")
print(output_path)


# ============================================================
# 15. SAVE HYBRID PREDICTIONS
# ============================================================

prediction_output = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(hybrid_predictions)
})

prediction_path = os.path.join(
    OUTPUT_DIR,
    "obesity_hybrid_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)

print("\nHybrid predictions saved to:")
print(prediction_path)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("HYBRID MODEL COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nProposed Model:")
print("Random Forest + MLP Probability Fusion")

print("\nComponents:")
print("1. Random Forest - Machine Learning branch")
print("2. MLP - Deep Learning branch")
print("3. Probability-level fusion")
print("4. Final hybrid prediction")

print("\nEvaluation Metrics:")
print("1. Accuracy")
print("2. Precision")
print("3. Recall / Sensitivity")
print("4. F1-score")
print("5. ROC-AUC")

print("\nNext step:")
print("Compare the hybrid model with the six existing models.")