
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"

FAILURE_MODEL_FILE = MODEL_DIR / "failure_model.joblib"
NOVELTY_MODEL_FILE = MODEL_DIR / "novelty_detector.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

CATEGORICAL_FEATURES = ["Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Machine failure"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. First run: python download_data.py"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Read the raw dataset directly. No separate cleaning step.
    df = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from CSV: {missing_columns}"
        )

    # Convert only the fields required by the model.
    # Invalid feature values become missing and are imputed by
    # the preprocessing pipeline. Invalid targets cannot be trained.
    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[TARGET] = df[TARGET].where(df[TARGET].isin([0, 1]))
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    if y.nunique() < 2:
        raise ValueError("The target must contain both 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    failure_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training failure prediction model...")
    failure_model.fit(X_train, y_train)

    predictions = failure_model.predict(X_test)
    probabilities = failure_model.predict_proba(X_test)[:, 1]

    metrics = {
        "rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "failure_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No failure", "Failure"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_test.nunique() == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, probabilities)
        )
        metrics["average_precision"] = float(
            average_precision_score(y_test, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    # Learn normal sensor patterns from the training split only.
    print("Training unusual-pattern detector...")
    X_train_transformed = failure_model.named_steps[
        "preprocessor"
    ].transform(X_train)

    novelty_detector = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
    )
    novelty_detector.fit(X_train_transformed)

    joblib.dump(failure_model, FAILURE_MODEL_FILE)
    joblib.dump(novelty_detector, NOVELTY_MODEL_FILE)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed successfully.")
    print("Failure model:", FAILURE_MODEL_FILE)
    print("Novelty detector:", NOVELTY_MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print("ROC AUC:", metrics["roc_auc"])
    print("Average precision:", metrics["average_precision"])

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No failure", "Failure"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
