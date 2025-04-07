from flask import Flask, render_template, request
import requests
import os
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1hsP79tcdDgTGqUjpsxYP6Tva9MvRr-tT"
MODEL_PATH = "models/flight_delay_model.pkl"

# Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")

download_model()

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Mapping for display
CARRIER_MAP = {
    0: "American Airlines", 1: "Delta", 2: "United", 3: "Southwest", 4: "JetBlue"
}
AIRPORT_MAP = {
    0: "JFK", 1: "LAX", 2: "ORD", 3: "ATL", 4: "DFW"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    category = None
    color = None
    timestamp = None
    contributing_factors = {}

    if request.method == "POST":
        carrier = int(request.form["carrier"])
        origin = int(request.form["origin"])
        destination = int(request.form["destination"])
        dep_hour = int(request.form["dep_hour"])

        features = np.array([[carrier, origin, destination, dep_hour]])
        delay_minutes = model.predict(features)[0]

        # Convert delay to probability & category
        if delay_minutes < 15:
            probability = 0.1
            category = "On Time"
            color = "green"
        elif delay_minutes < 45:
            probability = 0.5
            category = "Minor Delay"
            color = "yellow"
        else:
            probability = 0.9
            category = "Major Delay"
            color = "red"

        probability_percent = int(probability * 100)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        contributing_factors = {
            "Carrier": CARRIER_MAP.get(carrier, f"Carrier {carrier}"),
            "Origin": AIRPORT_MAP.get(origin, f"Airport {origin}"),
            "Destination": AIRPORT_MAP.get(destination, f"Airport {destination}"),
            "Departure Hour": f"{dep_hour}:00"
        }

        return render_template("index.html",
                               prediction=True,
                               probability=probability_percent,
                               category=category,
                               color=color,
                               timestamp=timestamp,
                               contributing_factors=contributing_factors)

    return render_template("index.html", prediction=False)

if __name__ == "__main__":
    app.run(debug=True)
