import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("C:/Users/Owner/OneDrive/Documents/Datasets/health care diabetes.csv")  # Ensure you have the dataset

# Split data into features (X) and target (y)
X = df.drop(columns=["Outcome"])  # Assuming 'Outcome' is the target column
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data (important for better model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, "diabetes_model.pkl")



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

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model files not found! Make sure 'diabetes_model.pkl' and 'scaler.pkl' are uploaded to GitHub.")
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


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