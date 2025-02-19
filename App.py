# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:04:24 2025

@author: Owner
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

if not os.path.exists(model) or not os.path.exists(scaler):
    st.error("Model files not found! Make sure 'diabetes_model.pkl' and 'scaler.pkl' are uploaded to GitHub.")
else:
    model = joblib.load(model)
    scaler = joblib.load(scaler)


# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter the patient details below to predict diabetes.")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 25)

# Predict function
if st.button("Predict"):
    # Create a DataFrame for model input
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    
    # Scale the input data
    user_data_scaled = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_data_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    # Display result
    st.success(f"Prediction: {result}")