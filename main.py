from flask import Flask, render_template, request, jsonify
import requests
import os
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = "flight_delay_model_bz2.pkl"
GOOGLE_DRIVE_FILE_ID = "1hsP79tcdDgTGqUjpsxYP6Tva9MvRr-tT"

def download_model_from_drive(drive_id, output_path):
    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(output_path) < 25 * 1024 * 1024:
        raise ValueError("❌ Downloaded file is too small. Likely incomplete.")

# Always redownload (or delete if exists)
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

print("📥 Downloading model from Google Drive...")
download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")


# === Label mappings ===
carrier_map = {"AA": 0, "DL": 1, "UA": 2}
origin_map = {"JFK": 0, "LAX": 1, "ORD": 2}
dest_map = {"ATL": 0, "DFW": 1, "DEN": 2}

# === Routes ===
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

        print(f"🚀 Received: carrier={carrier}, origin={origin}, destination={destination}, time={dep_time}")

        if not all([carrier, origin, destination, dep_time]):
            return jsonify({"error": "Missing input values"}), 400

        # Convert time (HH:MM) to minutes since midnight
        try:
            dep_dt = datetime.strptime(dep_time, "%H:%M")
            dep_minutes = dep_dt.hour * 60 + dep_dt.minute
        except ValueError:
            return jsonify({"error": "Invalid time format"}), 400

        # Map inputs to encoded values
        carrier_val = carrier_map.get(carrier, -1)
        origin_val = origin_map.get(origin, -1)
        dest_val = dest_map.get(destination, -1)

        if -1 in (carrier_val, origin_val, dest_val):
            return jsonify({"error": "Invalid carrier, origin, or destination"}), 400

        features = np.array([[carrier_val, origin_val, dest_val, dep_minutes]])
        predicted_delay = model.predict(features)[0]

        # Convert delay minutes to probability
        probability = min(max(predicted_delay / 120, 0), 1)
        category = "Low" if probability < 0.33 else "Medium" if probability < 0.66 else "High"

        return jsonify({
            "probability": round(probability * 100, 2),
            "category": category,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500


# === Uncomment below for local dev ===
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)

