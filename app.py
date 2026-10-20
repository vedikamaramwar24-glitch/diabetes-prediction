
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load our saved machine learning model
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get all 8 values entered in the webpage
    pregnancies = float(request.form.get("Pregnancies", 0))
    glucose = float(request.form["Glucose"])
    blood_pressure = float(request.form["BloodPressure"])
    skin_thickness = float(request.form["SkinThickness"])
    insulin = float(request.form["Insulin"])
    bmi = float(request.form["BMI"])
    diabetes_pedigree = float(request.form["DiabetesPedigreeFunction"])
    age = float(request.form["Age"])

    # Keep the inputs in the same order as model training
    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    # Ask the model to make a prediction
    result = model.predict(input_data)[0]

    if result == 1:
        message = "The model predicts a higher likelihood of diabetes."
    else:
        message = "The model predicts a lower likelihood of diabetes."

    return render_template("index.html", prediction=message)

if __name__ == "__main__":
    app.run(debug=True)
