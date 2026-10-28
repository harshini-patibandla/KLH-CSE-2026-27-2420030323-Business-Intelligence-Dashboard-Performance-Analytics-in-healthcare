
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

# ==========================================
# HOSPITAL HYBRID MODEL 2
# Random Forest + XGBoost Probability Fusion
# ==========================================

DATA_PATH = "outputs/results/hospital_feature_engineered.csv"
RESULTS_DIR = "outputs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Use admission-time features only
feature_columns = ["Age", "Gender", "Condition"]
target_column = "Outcome"

required_columns = feature_columns + [target_column]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df[required_columns].dropna().copy()

X = df[feature_columns]
y = df[target_column]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

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

# Random Forest pipeline
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# XGBoost pipeline
xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

# Obtain and combine probabilities
rf_prob = rf_model.predict_proba(X_test)
xgb_prob = xgb_model.predict_proba(X_test)

# Both models use the same encoded target classes
hybrid_prob = (0.5 * rf_prob) + (0.5 * xgb_prob)
y_pred = np.argmax(hybrid_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 55)
print("HOSPITAL HYBRID MODEL 2 — RESULTS")
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
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

# Save metrics
metrics_df = pd.DataFrame([{
    "Model": "RF + XGBoost Probability Fusion",
    "Features": ", ".join(feature_columns),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}])

metrics_path = os.path.join(
    RESULTS_DIR, "hospital_hybrid_model_2_results.csv"
)
metrics_df.to_csv(metrics_path, index=False)

# Save confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in label_encoder.classes_],
    columns=[f"Predicted_{c}" for c in label_encoder.classes_]
)

cm_path = os.path.join(
    RESULTS_DIR, "hospital_hybrid_model_2_confusion_matrix.csv"
)
cm_df.to_csv(cm_path)

# Save predictions
predictions_df = pd.DataFrame({
    "Actual_Outcome": label_encoder.inverse_transform(y_test),
    "Predicted_Outcome": label_encoder.inverse_transform(y_pred)
})

predictions_path = os.path.join(
    RESULTS_DIR, "hospital_hybrid_model_2_predictions.csv"
)
predictions_df.to_csv(predictions_path, index=False)

print("\nFiles saved:")
print(metrics_path)
print(cm_path)
print(predictions_path)
print("\nHybrid Model 2 completed.")
