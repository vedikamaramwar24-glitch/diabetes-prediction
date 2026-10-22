
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "Diabetics prediction.csv")
MODEL_FILE = os.path.join(BASE_DIR, "diabetes_model.pkl")

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]


def train_models():
    df = pd.read_csv(DATASET_FILE)
    df.columns = df.columns.str.strip()

    if "Outcome" not in df.columns:
        raise ValueError("Dataset must contain an Outcome column.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Outcome"], errors="coerce")

    valid = y.isin([0, 1])
    X, y = X.loc[valid], y.loc[valid].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Dataset needs at least two examples of each outcome.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(random_state=42)
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=100, random_state=42
            )
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "accuracy": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "precision": round(
                precision_score(y_test, predictions, zero_division=0) * 100, 2
            ),
            "recall": round(
                recall_score(y_test, predictions, zero_division=0) * 100, 2
            ),
            "f1": round(
                f1_score(y_test, predictions, zero_division=0) * 100, 2
            )
        }

    best_model_name = max(
        results, key=lambda name: results[name]["f1"]
    )

    joblib.dump({
        "models": models,
        "results": results,
        "best_model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS
    }, MODEL_FILE)

    print("Model comparison:")
    for name, scores in results.items():
        print(name, scores)

    print("Best model:", best_model_name)
    print("Saved model file:", MODEL_FILE)


if __name__ == "__main__":
    train_models()
