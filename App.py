import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib  # type: ignore # For loading the ML model
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load trained ML model (replace 'model.pkl' with actual model file)
@st.cache_data
def load_model():
    return joblib.load("best_classification_model.pkl")

model = load_model()

# Title
st.title("🚗 Road Accident Survival Prediction & Analysis")

# Sidebar inputs
st.sidebar.header("User Input Features")

road_type = st.sidebar.selectbox("Road Type", ["Highway", "City Road", "Rural Road"])
weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Snowy"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Car", "Bike", "Truck", "Bus"])
speed = st.sidebar.slider("Vehicle Speed (km/h)", 0, 200, 60)
seatbelt = st.sidebar.radio("Seatbelt Worn?", ["Yes", "No"])

# Convert categorical inputs to numerical
input_data = {
    "road_type": [road_type],
    "weather": [weather],
    "vehicle_type": [vehicle_type],
    "speed": [speed],
    "seatbelt": [1 if seatbelt == "Yes" else 0],
}

df = pd.DataFrame(input_data)

# Make prediction
if st.sidebar.button("Predict Survival Chance"):
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] * 100
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Survival Likely! ({probability:.2f}% chance)")
    else:
        st.error(f"High Risk! ({probability:.2f}% chance)")

# Visualization - Sample Accident Data
st.subheader("📊 Accident Data Analysis")
data = pd.read_csv("accident.csv")  # Replace with actual dataset
fig, ax = plt.subplots()
sns.countplot(x="road_type", data=data, ax=ax)
st.pyplot(fig)

st.write("Data Source: Your dataset")

# Run the app using `streamlit run script.py`
