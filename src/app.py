import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# Ensure these file names match exactly what you uploaded
MODEL_PATH = "src/model/random_forest_model.joblib"
FEATURES_PATH = "src/model/feature_columns.joblib"

# Preprocessing Constants (Derived from your training data)
MEDIAN_AGE = 28.0
AGE_UPPER_BOUND = 55.5
MEDIAN_FARE = 14.45

# Load Model and Feature Columns
try:
    model = joblib.load(MODEL_PATH)
    expected_features = joblib.load(FEATURES_PATH)
    print("Model and Features loaded successfully!")
    print(f"Expecting {len(expected_features)} features: {expected_features}")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    model = None
    expected_features = []

@app.route("/")
def home():
    # This looks for 'index.html' inside the 'templates' folder
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        # 1. Get Raw Data from Frontend
        data = request.get_json()
        
        # 2. Preprocess Data (Replicating your training logic)
        
        # --- Numeric Inputs ---
        pclass = int(data.get("Pclass", 3))
        sibsp = int(data.get("SibSp", 0))
        parch = int(data.get("Parch", 0))
        
        # --- Age: Impute & Cap ---
        raw_age = data.get("Age")
        age = float(raw_age) if raw_age not in [None, ""] else MEDIAN_AGE
        # Cap age at 55.5 (from your IQR analysis)
        age_capped = min(age, AGE_UPPER_BOUND)
        
        # --- Fare: Impute & Log Transform ---
        raw_fare = data.get("Fare")
        fare = float(raw_fare) if raw_fare not in [None, ""] else MEDIAN_FARE
        # Apply log(1+x)
        fare_log = np.log1p(fare)
        
        # --- Sex: Binary Encoding ---
        sex_str = str(data.get("Sex", "")).lower()
        sex_encoded = 1 if sex_str == "female" else 0
        
        # --- Embarked: One-Hot Encoding ---
        embarked = str(data.get("Embarked", "S")).upper()
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0
        
        # --- Cabin: Feature Engineering ---
        cabin_str = str(data.get("Cabin", "")).strip()
        has_cabin = 1 if cabin_str else 0
        
        # Extract Deck (First letter)
        deck_letter = cabin_str[0].upper() if has_cabin else "U"
        
        # 3. Build the Feature Vector
        # We create a dictionary to map calculated values to the expected column names
        features_dict = {
            'Pclass': pclass,
            'SibSp': sibsp,
            'Parch': parch,
            'Has_Cabin': has_cabin,
            'Age_capped': age_capped,
            'Fare_log': fare_log,
            'Sex_encoded': sex_encoded,
            'Embarked_Q': embarked_q,
            'Embarked_S': embarked_s,
            # Decks: Logic to set the specific deck bit to 1
            f'Deck_{deck_letter}': 1 
        }

        # Create the final ordered list based on 'feature_columns.joblib'
        # If a deck column (e.g., Deck_B) is not in our dict, it defaults to 0
        final_input = [features_dict.get(col, 0) for col in expected_features]
        
        # Convert to 2D array for prediction
        features_np = np.array([final_input])
        
        # 4. Make Prediction
        probability = model.predict_proba(features_np)[0][1]
        prediction = "Survived" if probability > 0.5 else "Did Not Survive"
        
        return jsonify({
            "prediction": prediction,
            "probability": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)