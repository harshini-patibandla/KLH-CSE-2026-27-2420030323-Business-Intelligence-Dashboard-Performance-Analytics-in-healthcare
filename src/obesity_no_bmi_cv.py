# ============================================================
# OBESITY NO-BMI EXPERIMENT
# Repeated Stratified 5-Fold Cross-Validation
# Healthcare BI Dashboard
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results"
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "obesity_feature_engineered.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "obesity_no_bmi_repeated_cv.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("OBESITY NO-BMI EXPERIMENT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 3. REMOVE BMI / ANTHROPOMETRIC FEATURES
# ============================================================

target = "NObeyesdad"

# Remove features directly related to BMI/obesity labelling
columns_to_remove = []

for column in df.columns:

    name = column.lower()

    if (
        name == "bmi"
        or name.startswith("bmi_")
        or "bmi_category" in name
        or "obesity_signal" in name
        or name == "height"
        or name == "weight"
    ):
        columns_to_remove.append(column)


# Never remove target
columns_to_remove = [
    c for c in columns_to_remove
    if c != target
]


print("\nFeatures removed:")
for column in sorted(columns_to_remove):
    print(" -", column)


# ============================================================
# 4. CREATE X AND Y
# ============================================================

X = df.drop(
    columns=[target] + columns_to_remove
)

y = df[target].astype(str)

print("\nRemaining features:")
print(list(X.columns))

print("\nNumber of remaining features:", X.shape[1])
print("Number of samples:", X.shape[0])


# ============================================================
# 5. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget classes:")
print(list(label_encoder.classes_))


# ============================================================
# 6. IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["int64", "float64"]
).columns.tolist()


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 8. MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="mlogloss"
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=100,
            verbose=0,
            random_state=42
        ),

    "MLP":
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        )
}


# ============================================================
# 9. REPEATED STRATIFIED 5-FOLD CV
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)


scoring = {

    "accuracy": "accuracy",

    "precision": make_scorer(
        precision_score,
        average="weighted",
        zero_division=0
    ),

    "recall": make_scorer(
        recall_score,
        average="weighted",
        zero_division=0
    ),

    "f1": make_scorer(
        f1_score,
        average="weighted",
        zero_division=0
    )
}


# ============================================================
# 10. RUN ALL MODELS
# ============================================================

all_results = []

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print("Running:", model_name)
    print("-" * 70)

    pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    scores = cross_validate(
        pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    result = {

        "Model": model_name,

        "Accuracy_Mean":
            scores["test_accuracy"].mean(),

        "Accuracy_STD":
            scores["test_accuracy"].std(),

        "Precision_Mean":
            scores["test_precision"].mean(),

        "Precision_STD":
            scores["test_precision"].std(),

        "Recall_Mean":
            scores["test_recall"].mean(),

        "Recall_STD":
            scores["test_recall"].std(),

        "F1_Mean":
            scores["test_f1"].mean(),

        "F1_STD":
            scores["test_f1"].std()
    }

    all_results.append(result)

    print(
        f"Accuracy : "
        f"{result['Accuracy_Mean']:.4f} "
        f"+/- {result['Accuracy_STD']:.4f}"
    )

    print(
        f"Precision: "
        f"{result['Precision_Mean']:.4f} "
        f"+/- {result['Precision_STD']:.4f}"
    )

    print(
        f"Recall   : "
        f"{result['Recall_Mean']:.4f} "
        f"+/- {result['Recall_STD']:.4f}"
    )

    print(
        f"F1       : "
        f"{result['F1_Mean']:.4f} "
        f"+/- {result['F1_STD']:.4f}"
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="F1_Mean",
    ascending=False
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 12. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL NO-BMI RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("NO-BMI EXPERIMENT COMPLETED")
print("=" * 70)