import streamlit as st
import pandas as pd
import joblib
import numpy as np
from utils.preprocess import preprocess_input

# Load model and preprocessing tools
model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")

label_encoders = {
    "Sex": joblib.load("model/le_Sex.pkl"),
    "Housing": joblib.load("model/le_Housing.pkl"),
    "Saving accounts": joblib.load("model/le_Saving accounts.pkl"),
    "Checking account": joblib.load("model/le_Checking account.pkl"),
    "Purpose": joblib.load("model/le_Purpose.pkl"),
}

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.markdown("Predict if a person has a **High Credit Risk** based on input data.")

# User input
def user_input():
    age = st.slider("Age", 18, 75, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.slider("Job", 0, 3, 2)
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
    credit_duration = st.slider("Duration (months)", 4, 72, 24)
    purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])
    credit_amount = st.number_input("Credit Amount", min_value=100, max_value=20000, value=1000)

    data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Duration": credit_duration,
        "Purpose": purpose,
        "Credit amount": credit_amount,
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input()

if st.button("Predict Credit Risk"):
    processed_input = preprocess_input(df_input, label_encoders, scaler)
    prediction = model.predict(processed_input)[0]

    st.subheader("Prediction Result")
    # st.success("✅ Low Credit Risk") if prediction == 0 else st.error("❌ High Credit Risk")

    if prediction == 0:
        st.markdown("<h1 style='color: green;'>✅ Low Credit Risk</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='color: red;'>❌ High Credit Risk</h1>", unsafe_allow_html=True)
