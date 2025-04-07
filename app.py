from flask import Flask, render_template, request, jsonify
import gdown
import os
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = "flight_delay_model_bz2.pkl"
GOOGLE_DRIVE_FILE_ID = "1hsP79tcdDgTGqUjpsxYP6Tva9MvRr-tT"
MODEL_URL = f"https://drive.google.com/uc?id=1hsP79tcdDgTGqUjpsxYP6Tva9MvRr-tT"

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Example label encodings (replace with yours if different)
carrier_map = {"AA": 0, "DL": 1, "UA": 2}
origin_map = {"JFK": 0, "LAX": 1, "ORD": 2}
dest_map = {"ATL": 0, "DFW": 1, "DEN": 2}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        carrier = request.form.get("carrier")
        origin = request.form.get("origin")
        destination = request.form.get("destination")
        dep_time = request.form.get("departure_time")

        dep_minutes = datetime.strptime(dep_time, "%H:%M").hour * 60 + datetime.strptime(dep_time, "%H:%M").minute

        carrier_val = carrier_map.get(carrier, -1)
        origin_val = origin_map.get(origin, -1)
        dest_val = dest_map.get(destination, -1)

        if -1 in (carrier_val, origin_val, dest_val):
            return jsonify({"error": "Invalid input"}), 400

        features = np.array([[carrier_val, origin_val, dest_val, dep_minutes]])
        predicted_delay = model.predict(features)[0]

        probability = min(max(predicted_delay / 120, 0), 1)
        category = "Low" if probability < 0.33 else "Medium" if probability < 0.66 else "High"

        return jsonify({
            "probability": round(probability * 100, 2),
            "category": category,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
