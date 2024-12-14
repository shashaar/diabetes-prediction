from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model_path = "model/trained_model.joblib"
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None  # Initialize result to None to avoid displaying previous prediction on reload

    if request.method == "POST":
        # Get user inputs and map to the correct feature names
        data = {
            "Pregnancies Month": [int(request.form["pregnancies"])],
            "Glucose (mg)": [int(request.form["glucose"])],
            "Blood Pressure (mmHg)": [int(request.form["blood_pressure"])],
            "Skin Thickness (mm)": [int(request.form["skin_thickness"])],
            "Insulin (pmol/L)": [int(request.form["insulin"])],
            "BMI": [float(request.form["bmi"])],
            "Diabetes Pedigree Function (e.g 0.49)": [float(request.form["diabetes_pedigree"])],
            "Age": [int(request.form["age"])],
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(data)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # Prepare the result
        result = {
            "prediction": "POSITIVE" if prediction == 1 else "NEGATIVE",
            "probability_negative": f"{probability[0] * 100:.2f}%",
            "probability_positive": f"{probability[1] * 100:.2f}%",
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
