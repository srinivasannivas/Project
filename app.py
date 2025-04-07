from flask import Flask, render_template, request
import numpy as np
import pickle
from datetime import datetime
import requests

app = Flask(__name__)

# Load model from Dropbox
DROPBOX_URL = "https://drive.google.com/file/d/1hsP79tcdDgTGqUjpsxYP6Tva9MvRr-tT/view?usp=sharing"

try:
    response = requests.get(DROPBOX_URL)
    response.raise_for_status()
    model = pickle.loads(response.content)
except Exception as e:
    raise RuntimeError(f"Failed to load model from Dropbox: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve form inputs
        flight_date = request.form["flight_date"]
        carrier = int(request.form["carrier"])
        origin = int(request.form["origin"])
        destination = int(request.form["destination"])
        crs_dep_time = int(request.form["crs_dep_time"])

        # Parse flight date
        flight_datetime = datetime.strptime(flight_date, "%Y-%m-%d")
        month = flight_datetime.month - 1
        day = flight_datetime.day - 1
        weekday = flight_datetime.isoweekday() - 1

        # Build feature array
        features = np.array([[month, day, weekday, carrier, origin, destination, crs_dep_time, 0]])

        # Predict delay in minutes
        prediction_minutes = max(round(model.predict(features)[0], 2), 0)

        # Determine delay category
        if prediction_minutes < 15:
            delay_category = "Low"
        elif prediction_minutes < 45:
            delay_category = "Medium"
        else:
            delay_category = "High"

        # Compute delay probability (mock logic)
        probability = min(int((prediction_minutes / 60) * 100), 100)

        # Placeholder contributing factors (could be improved with feature importance)
        contributing_factors = ["Weather", "Air Traffic", "Airline History"]

        return render_template("index.html",
                               prediction_text=f"Estimated delay: {prediction_minutes} minutes",
                               delay_category=delay_category,
                               probability=probability,
                               timestamp=datetime.now().strftime("%I:%M:%S %p"),
                               contributing_factors=contributing_factors)
    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
