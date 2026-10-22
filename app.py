
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "diabetes_models.pkl")

try:
    data = joblib.load(MODEL_FILE)

    models = data["models"]
    best_model_name = data["best_model_name"]
    FEATURE_COLUMNS = data["feature_columns"]

    comparison_results = [
        {"name": name, **scores}
        for name, scores in data["results"].items()
    ]

    startup_error = None

except Exception as error:
    models = {}
    best_model_name = None
    FEATURE_COLUMNS = []
    comparison_results = []
    startup_error = (
        "Could not load the trained models. "
        "Check diabetes_model.pkl and its format. "
        f"Details: {error}"
    )


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction=None,
        comparison_results=comparison_results,
        best_model_name=best_model_name,
        startup_error=startup_error,
        selected_model=best_model_name
    )


@app.route("/predict", methods=["POST"])
def predict():
    if startup_error or not models:
        return render_template(
            "index.html",
            prediction=None,
            comparison_results=comparison_results,
            best_model_name=best_model_name,
            startup_error=startup_error,
            selected_model=best_model_name
        )

    try:
        values = {}

        for column in FEATURE_COLUMNS:
            raw = request.form.get(column, "").strip()

            if not raw:
                raise ValueError(f"Please enter a value for {column}.")

            value = float(raw)

            if not pd.notna(value) or value < 0:
                raise ValueError(
                    f"Please enter a valid non-negative value for {column}."
                )

            values[column] = value

        input_data = pd.DataFrame(
            [values], columns=FEATURE_COLUMNS
        )

        selected_model = models[best_model_name]
        result = int(selected_model.predict(input_data)[0])

        if result == 1:
            prediction_text = (
                "The model predicts a diabetes-positive outcome "
                "(Outcome = 1)."
            )
        else:
            prediction_text = (
                "The model predicts a diabetes-negative outcome "
                "(Outcome = 0)."
            )

        error = None

    except (ValueError, TypeError) as exc:
        prediction_text = None
        error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction_text,
        comparison_results=comparison_results,
        best_model_name=best_model_name,
        startup_error=error,
        selected_model=best_model_name
    )


if __name__ == "__main__":
    app.run(debug=True)
