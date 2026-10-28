
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Paths
input_path = "outputs/results/healthcare_selected_features.csv"
results_dir = "outputs/results"
os.makedirs(results_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_path)
target = "Test Results"
df = df.dropna(subset=[target]).copy()

X = pd.get_dummies(df.drop(columns=[target]), drop_first=False)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)

encoder = LabelEncoder()
y = encoder.fit_transform(df[target].astype(str))

# Same split settings as Hybrid Model 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Model 2: MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# Equal-weight probability fusion
rf_probs = rf.predict_proba(X_test)
mlp_probs = mlp.predict_proba(X_test_scaled)

hybrid_probs = 0.5 * rf_probs + 0.5 * mlp_probs
predictions = np.argmax(hybrid_probs, axis=1)

# Evaluation
results = {
    "Dataset": "Healthcare",
    "Model": "RF + MLP Probability Fusion",
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(
        y_test, predictions, average="weighted", zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, average="weighted", zero_division=0
    ),
    "F1_Score": f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )
}

print("\nHEALTHCARE HYBRID MODEL 2")
print("=" * 45)
for key, value in results.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Save metrics
pd.DataFrame([results]).to_csv(
    os.path.join(results_dir, "healthcare_hybrid_model_2_results.csv"),
    index=False
)

# Save confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test, predictions, labels=np.arange(len(encoder.classes_))
)
pd.DataFrame(
    cm, index=encoder.classes_, columns=encoder.classes_
).to_csv(
    os.path.join(results_dir, "healthcare_hybrid_model_2_confusion_matrix.csv")
)

# Save predictions
pd.DataFrame({
    "Actual": encoder.inverse_transform(y_test),
    "Predicted": encoder.inverse_transform(predictions)
}).to_csv(
    os.path.join(results_dir, "healthcare_hybrid_model_2_predictions.csv"),
    index=False
)

print("\nResults, confusion matrix, and predictions saved.")
