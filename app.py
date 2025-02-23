import streamlit as st
import joblib
import numpy as np

# Load the trained SVM model and scaler
model = joblib.load("new_model.pkl")
scaler = joblib.load("scaler.pkl")  # Load the scaler used during training

# Streamlit UI
st.title("Heart Attack Risk Prediction")
st.write("Enter your health details to predict the risk of a heart attack.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=40)
sysBP = st.number_input("Systolic Blood Pressure (sysBP)", min_value=80, max_value=200, value=120)
diaBP = st.number_input("Diastolic Blood Pressure (diaBP)", min_value=50, max_value=130, value=80)
totChol = st.number_input("Total Cholesterol (totChol)", min_value=100, max_value=400, value=200)
glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
male = int(st.radio("Gender", ["Female", "Male"]) == "Male")
diabetes = int(st.radio("Diabetes", ["No", "Yes"]) == "Yes")
Hypertension = int(st.radio("Hypertension", ["No", "Yes"]) == "Yes")

# Convert inputs to array and scale
features = np.array([[age, sysBP, diaBP, totChol, glucose, BMI, male, diabetes, Hypertension]])
features_scaled = scaler.transform(features)  # Apply the same scaling as training

# Prediction
if st.button("Predict Heart Attack Risk"):
    prediction = model.predict(features_scaled)
    result = "⚠️ High Risk" if prediction[0] == 1 else "✅ Low Risk"
    st.subheader(f"Prediction: {result}")
