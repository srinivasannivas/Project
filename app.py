from flask import Flask, render_template, request, jsonify
import pickle
import requests
import io
import os
from datetime import datetime

app = Flask(__name__)

# Your Google Drive file ID
MODEL_FILE_ID = "1hsP79tcdDgTGqUjpsxYP6Tva9MvRr-tT"

def download_model():
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
    response = requests.get(url)

    content_type = response.headers.get("Content-Type", "")
    if "html" in content_type:
        raise ValueError("Google Drive returned an HTML page instead of the model file. Make sure the file is public.")

    try:
        model = pickle.load(io.BytesIO(response.content))
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from Google Drive: {e}")

# Load the model once at startup
model = download_model()

# Sample feature names for dummy inputs — adjust as needed
feature_names = ['carrier', 'origin', 'dest', 'sched_dep_hour']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract input features from request JSON
        input_data = [
            data.get('carrier', 0),
            data.get('origin', 0),
            data.get('dest', 0),
            data.get('sched_dep_hour', 0)
        ]

        # Run prediction
        predicted_minutes = model.predict([input_data])[0]
        delay_probability = min(predicted_minutes / 120, 1.0)  # normalize up to 120 mins

        # Get delay category
        if predicted_minutes > 60:
            delay_category = "High Delay"
        elif predicted_minutes > 15:
            delay_category = "Moderate Delay"
        else:
            delay_category = "On Time"

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'success': True,
            'probability': round(delay_probability * 100, 2),
            'category': delay_category,
            'predicted_minutes': round(predicted_minutes),
            'timestamp': timestamp
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
