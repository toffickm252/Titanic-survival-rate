import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import logging

# Constants from your data cleaning script
AGE_UPPER_BOUND = 55.5

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "..", "model", "random_forest_model.pkl"))
    feature_path = os.getenv("FEATURE_PATH", os.path.join(base_dir, "..", "model", "feature_columns.pkl"))

    try:
        model = joblib.load(model_path)
        features = joblib.load(feature_path)
        return model, features
    except Exception as e:
        logging.error(f"Error loading artifacts: {e}")
        st.error("Failed to load model artifacts. Please check the logs.")
        st.stop()


model, feature_columns = load_artifacts()

st.title("Titanic Survival Prediction")

# --- User Inputs ---
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, value=0, step=1)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, value=14.45, step=1.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
cabin = st.text_input("Cabin (e.g., C85, or leave blank if unknown)")

# --- Prediction Logic ---
if st.button("Predict"):
    # --- 1. Preprocessing ---
    try:
        # Create a dictionary to hold all feature values, initialized to 0
        feature_dict = {col: 0 for col in feature_columns}

        # --- 2. Populate features from user input ---

        # Basic numeric features
        feature_dict['Pclass'] = pclass
        feature_dict['SibSp'] = sibsp
        feature_dict['Parch'] = parch

        # Processed numeric features
        feature_dict['Age_capped'] = min(age, AGE_UPPER_BOUND)
        feature_dict['Fare_log'] = np.log1p(fare)

        # Encoded categorical features
        feature_dict['Sex_encoded'] = 1 if sex == "female" else 0

        # Embarked (One-Hot Encoded)
        if f"Embarked_{embarked}" in feature_dict:
            feature_dict[f"Embarked_{embarked}"] = 1

        # Cabin and Deck features
        has_cabin = 1 if cabin.strip() else 0
        feature_dict['Has_Cabin'] = has_cabin
        if has_cabin:
            deck_letter = cabin.strip()[0].upper()
            deck_feature = f"Deck_{deck_letter}"
            if deck_feature in feature_dict:
                feature_dict[deck_feature] = 1

        # --- 3. Create DataFrame ---
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([feature_dict])[feature_columns]

        # --- 4. Make Prediction ---
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        label = "Survived" if pred == 1 else "Did Not Survive"

        st.success(label)
        st.info(f"Survival probability: {prob*100:.2f}%")

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("An error occurred during prediction. Please check the input values and try again.")
        st.stop()