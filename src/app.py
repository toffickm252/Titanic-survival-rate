# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature columns
model = joblib.load('model/random_forest_new_model.joblib')
feature_columns = joblib.load('model/feature_new_columns.joblib')

# Streamlit app

# Passenger Class input
st.title("Titanic Survival Prediction")
PassengerClass = st.selectbox("Pclass", [1, 2, 3])

st.write(f"You selected: {PassengerClass}")

# Sex input
# 1. Create the dropdown
sex_input = st.selectbox("Sex", ["Male", "Female"])

# 2. Translate to numbers
if sex_input == "Male":
    sex_encoded = 1
else:
    sex_encoded = 0

st.write(f"You selected: {sex_encoded}")

# Has_Cabin input
# 1. Create the dropdown
cabin_input = st.selectbox("Has_Cabin?", ["Yes", "No"])

# 2. Translate to numbers
if cabin_input == "Yes":
    has_cabin = 1
else:
    has_cabin = 0

st.write(f"You selected: {has_cabin}")

# SibSp input
SibSp = st.number_input("SibSp", min_value=0, max_value=8, value=0)

st.write(f"You entered: {SibSp}")

# Parch input
Parch = st.number_input("Parch", min_value=0, max_value=6, value=0)

st.write(f"You entered: {Parch}")



# Age_capped input
Age_input = st.slider("Age_capped", 0.0,100.0, 30.0)

# Cap the age at 80
age_cap = min(Age_input, 80)

st.write(f"You entered: {age_cap}")

# Fare_log input
fare_input = st.number_input("Fare_log", min_value=0.0, value=0.0, step=0.01)
Fare = np.log1p(fare_input)

st.write(f"You entered: {Fare}")

# 2. Pack all variables into a list in the exact order the model expects
input_data = [PassengerClass, SibSp, Parch, has_cabin, age_cap, Fare, sex_encoded]

# Make the prediction
prediction = model.predict([input_data])

# Create a button to trigger the prediction
if st.button("Predict"):
    prediction = model.predict([input_data])
    
    # Check the first item in the result (0 or 1)
    if prediction[0] == 1:
        st.success("The model predicts: Passenger Survived!")
    else:
        st.error("The model predicts: Passenger Did Not Survive")






