import streamlit as st
import joblib
import pandas as pd
import os

# Define paths for models
MODEL_PATHS = {
    "Heart Disease": "models/heart_disease_model.pkl",
    "Diabetes": "models/diabetes_model.pkl",
    "Parkinson's Disease": "models/parkinson_model.pkl"
}

# Function to load model safely
def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found: {path}")
        return None

# Streamlit UI
st.title("AI-Powered Medical Diagnosis System 🏥")

# Select Disease Model
disease = st.selectbox("Select the Disease for Prediction:", list(MODEL_PATHS.keys()))

# Load selected model
model = load_model(MODEL_PATHS[disease])

if model:
    st.success(f"Model for {disease} loaded successfully ✅")

    # Input Fields
    st.subheader(f"Enter the details for {disease} Prediction:")

    # Heart Disease Inputs (All Features Included)
    if disease == "Heart Disease":
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, step=1)
        chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, step=1)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, step=1)
        exang = st.selectbox("Exercise-Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1)
        slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, step=1)
        ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1)
        thal = st.number_input("Thalassemia Type (0-3)", min_value=0, max_value=3, step=1)

        # Make prediction
        if st.button("Predict Heart Disease"):
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            if input_data.shape[1] == 13:  
                prediction = model.predict(input_data)
                st.success("Heart Disease Detected ✅" if prediction[0] == 1 else "No Heart Disease ❌")
            else:
                st.error("Invalid input data shape for Heart Disease prediction.")

    # Diabetes Inputs (All Features Included)
    elif disease == "Diabetes":
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose Level", min_value=50, max_value=250, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=150, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
        age = st.number_input("Age", min_value=0, max_value=120, step=1)

        # Make prediction
        if st.button("Predict Diabetes"):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            if input_data.shape[1] == 8:  
                prediction = model.predict(input_data)
                st.success("Diabetes Detected ✅" if prediction[0] == 1 else "No Diabetes ❌")
            else:
                st.error("Invalid input data shape for Diabetes prediction.")

    # Parkinson’s Disease Inputs (All Features Included)
    elif disease == "Parkinson's Disease":
        parkinson_features = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", 
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", 
            "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", 
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]

        parkinson_inputs = [st.number_input(feature, step=0.01) for feature in parkinson_features]

        # Make prediction
        if st.button("Predict Parkinson’s Disease"):
            input_data = pd.DataFrame([parkinson_inputs])
            if input_data.shape[1] == 22:  
                prediction = model.predict(input_data)
                st.success("Parkinson’s Disease Detected ✅" if prediction[0] == 1 else "No Parkinson’s Disease ❌")
            else:
                st.error("Invalid input data shape for Parkinson's Disease prediction.")
