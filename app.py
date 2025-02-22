import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("new_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the app
st.title("Heart Attack Risk Predictor")

# User input fields
male = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, step=0.1)
sysBP = st.number_input("Systolic Blood Pressure", min_value=80.0, max_value=200.0, step=0.1)
diaBP = st.number_input("Diastolic Blood Pressure", min_value=50.0, max_value=130.0, step=0.1)
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, step=0.1)
glucose = st.number_input("Glucose Level", min_value=50.0, max_value=300.0, step=0.1)

# Convert categorical inputs to numerical
male = 1 if male == "Male" else 0
hypertension = 1 if hypertension == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Predict button
if st.button("Predict Heart Attack Risk"):
    features = np.array([[male, age, hypertension, diabetes, totChol, sysBP, diaBP, BMI, glucose]])
    prediction = model.predict(features)
    result = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.write(f"### Prediction: {result}")
