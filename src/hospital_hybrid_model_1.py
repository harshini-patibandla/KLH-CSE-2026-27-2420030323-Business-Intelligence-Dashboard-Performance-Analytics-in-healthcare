
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# HOSPITAL HYBRID MODEL 1
# Random Forest + MLP Probability Fusion
# Predict outcome using admission-time features
# ==========================================

DATA_PATH = "outputs/results/hospital_feature_engineered.csv"
RESULTS_DIR = "outputs/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Use only information available at admission.
# Exclude treatment, discharge, cost, satisfaction,
# and other information that may be known only later.
feature_columns = ["Age", "Gender", "Condition"]
target_column = "Outcome"

required_columns = feature_columns + [target_column]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df[required_columns].dropna().copy()

X = df[feature_columns]
y = df[target_column]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

numeric_features = ["Age"]
categorical_features = ["Gender", "Condition"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

# Model 1: Random Forest
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Model 2: Multilayer Perceptron
mlp_preprocessor = preprocessor

mlp_model = Pipeline([
    ("preprocessor", mlp_preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Training MLP...")
mlp_model.fit(X_train, y_train)

# Align class-probability columns before combining.
rf_prob = rf_model.predict_proba(X_test)
mlp_prob = mlp_model.predict_proba(X_test)

rf_classes = rf_model.named_steps["classifier"].classes_
mlp_classes = mlp_model.named_steps["classifier"].classes_
all_classes = np.arange(len(encoder.classes_))

def align_probabilities(probabilities, model_classes):
    aligned = np.zeros((probabilities.shape[0], len(all_classes)))
    for column, class_label in enumerate(model_classes):
        aligned[:, int(class_label)] = probabilities[:, column]
    return aligned

rf_prob = align_probabilities(rf_prob, rf_classes)
mlp_prob = align_probabilities(mlp_prob, mlp_classes)

# Hybrid: equal-weight probability fusion
hybrid_prob = (0.5 * rf_prob) + (0.5 * mlp_prob)
y_pred = np.argmax(hybrid_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 55)
print("HOSPITAL HYBRID MODEL 1 — RESULTS")
print("=" * 55)
print("Features:", feature_columns)
print("Target:", target_column)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=all_classes,
    target_names=encoder.classes_,
    zero_division=0
))

# Save metrics
metrics_df = pd.DataFrame([{
    "Model": "RF + MLP Probability Fusion",
    "Features": ", ".join(feature_columns),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}])

metrics_path = os.path.join(
    RESULTS_DIR, "hospital_hybrid_model_1_results.csv"
)
metrics_df.to_csv(metrics_path, index=False)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=all_classes)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in encoder.classes_],
    columns=[f"Predicted_{c}" for c in encoder.classes_]
)
cm_path = os.path.join(
    RESULTS_DIR, "hospital_hybrid_model_1_confusion_matrix.csv"
)
cm_df.to_csv(cm_path)

# Save predictions for inspection
predictions_df = pd.DataFrame({
    "Actual_Outcome": encoder.inverse_transform(y_test),
    "Predicted_Outcome": encoder.inverse_transform(y_pred)
})
predictions_path = os.path.join(
    RESULTS_DIR, "hospital_hybrid_model_1_predictions.csv"
)
predictions_df.to_csv(predictions_path, index=False)

print("\nFiles saved:")
print(metrics_path)
print(cm_path)
print(predictions_path)
print("\nHybrid Model 1 completed.")
