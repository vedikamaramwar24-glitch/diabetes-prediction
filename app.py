from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

DATASET_FILE = "Diabetics prediction.csv"

# These are the input columns used by the website and the usual diabetes dataset.
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_and_train_models():
    """Load the dataset, train three models, and compare them on one test split."""
    df = pd.read_csv(DATASET_FILE)
    df.columns = df.columns.str.strip()

    if "Outcome" not in df.columns:
        raise ValueError(
            "The dataset must contain an 'Outcome' column. "
            "Check the CSV file's column names."
        )

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(missing_columns)
        )

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Outcome"], errors="coerce")

    # Remove rows where the target Outcome is missing or invalid.
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
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
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
    }

    trained_models = {}
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "name": name,
            "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
            "precision": round(
                precision_score(y_test, predictions, pos_label=1, zero_division=0) * 100, 2
            ),
            "recall": round(
                recall_score(y_test, predictions, pos_label=1, zero_division=0) * 100, 2
            ),
            "f1": round(
                f1_score(y_test, predictions, pos_label=1, zero_division=0) * 100, 2
            ),
        })
        trained_models[name] = model

    # For this educational comparison, select the model with the highest
    # F1-score for Outcome=1. Check recall as well when discussing results.
    best_result = max(results, key=lambda row: row["f1"])
    best_model_name = best_result["name"]

    return trained_models, results, best_model_name


try:
    models, comparison_results, best_model_name = load_and_train_models()
    startup_error = None
except Exception as error:
    models, comparison_results, best_model_name = {}, [], None
    startup_error = str(error)


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        comparison_results=comparison_results,
        best_model_name=best_model_name,
        startup_error=startup_error,
        selected_model=best_model_name,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if startup_error or not models:
        return render_template(
            "index.html",
            prediction=None,
            comparison_results=comparison_results,
            best_model_name=best_model_name,
            startup_error=startup_error or "The models could not be trained.",
            selected_model=best_model_name,
        )

    try:
        values = {}
        for column in FEATURE_COLUMNS:
            raw_value = request.form.get(column, "").strip()
            if raw_value == "":
                raise ValueError(f"Please enter a value for {column}.")
            values[column] = float(raw_value)

        input_data = pd.DataFrame([values], columns=FEATURE_COLUMNS)

        # The website uses the model with the highest test-set F1-score.
        selected_model = models[best_model_name]
        result = int(selected_model.predict(input_data)[0])

        if result == 1:
            prediction_text = (
                "The model predicts a diabetes-positive outcome (Outcome = 1)."
            )
        else:
            prediction_text = (
                "The model predicts a diabetes-negative outcome (Outcome = 0)."
            )

        return render_template(
            "index.html",
            prediction=prediction_text,
            comparison_results=comparison_results,
            best_model_name=best_model_name,
            startup_error=None,
            selected_model=best_model_name,
        )

    except ValueError as error:
        return render_template(
            "index.html",
            prediction=None,
            comparison_results=comparison_results,
            best_model_name=best_model_name,
            startup_error=str(error),
            selected_model=best_model_name,
        )


if __name__ == "__main__":
    app.run(debug=True)
