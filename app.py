# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load Model, Scaler, and Encoders ---
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/le_education.pkl', 'rb') as f:
        le_education = pickle.load(f)
    with open('models/le_self_employed.pkl', 'rb') as f:
        le_self_employed = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval. This app uses a model trained on the dataset you provided.")

# --- Input Form ---
with st.form("loan_application_form"):
    st.subheader("Applicant Financial Details")
    
    col1, col2 = st.columns(2)
    with col1:
        income_annum = st.number_input("Annual Income ($)", min_value=0)
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0)
        loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30)

    with col2:
        education = st.selectbox("Education", options=le_education.classes_)
        self_employed = st.selectbox("Self Employed", options=le_self_employed.classes_)
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)

    st.subheader("Applicant Asset Details ($)")
    col3, col4 = st.columns(2)
    with col3:
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
    with col4:
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

    submitted = st.form_submit_button("Predict Loan Approval")

# --- Prediction Logic ---
if submitted:
    # Transform categorical inputs
    education_encoded = le_education.transform([education])[0]
    self_employed_encoded = le_self_employed.transform([self_employed])[0]

    # Create a feature array in the correct order
    input_features = np.array([[
        no_of_dependents, education_encoded, self_employed_encoded, income_annum,
        loan_amount, loan_term, cibil_score, residential_assets_value,
        commercial_assets_value, luxury_assets_value, bank_asset_value
    ]])
    
    # Scale the features
    input_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # loan_status mapping: 0 = Approved, 1 = Rejected in this dataset
    is_approved = prediction[0] == 0

    st.subheader("Prediction Result")
    if is_approved:
        st.success(f"**Loan Approved** ✔️ (Confidence: {prediction_proba[0][0]*100:.2f}%)")
        st.balloons()
    else:
        st.error(f"**Loan Rejected** ❌ (Confidence: {prediction_proba[0][1]*100:.2f}%)")