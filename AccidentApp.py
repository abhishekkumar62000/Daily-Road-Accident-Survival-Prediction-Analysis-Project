import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("best_classification_model.pkl")

# Title
st.title("Daily Road Accident Survival Prediction")

# User Inputs
st.sidebar.header("Enter Accident Details")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    helmet_used = st.sidebar.selectbox("Helmet Used", ["Yes", "No"])
    seatbelt_used = st.sidebar.selectbox("Seatbelt Used", ["Yes", "No"])
    speed_of_impact = st.sidebar.slider("Speed of Impact (km/h)", 0, 200, 50)
    
    data = {
        "Age": age,
        "Gender": gender,
        "Helmet_Used": helmet_used,
        "Seatbelt_Used": seatbelt_used,
        "Speed_of_Impact": speed_of_impact
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_df)
    st.write(f"### Prediction: {'Survived' if prediction[0] == 1 else 'Not Survived'}")
