# ==========================================
# CREDIT CARD FRAUD DETECTION - STREAMLIT APP
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Fraud Detection System", page_icon="💳")

st.title("💳 Credit Card Fraud Detection System")
st.write("Upload transaction data to detect fraudulent transactions")


# ==========================================
# LOAD MODEL + SCALER
# ==========================================

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())


    # ==========================================
    # DROP UNUSED COLUMNS (SAME AS TRAINING)
    # ==========================================

    drop_cols = [
        'Unnamed: 0',
        'trans_date_trans_time',
        'cc_num',
        'merchant',
        'first',
        'last'
    ]

    for col in drop_cols:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)


    # ==========================================
    # ❗ DROP TARGET COLUMN IF PRESENT
    # ==========================================

    if 'is_fraud' in data.columns:
        data.drop('is_fraud', axis=1, inplace=True)


    # ==========================================
    # ENCODE CATEGORICAL COLUMNS
    # ==========================================

    encoder = LabelEncoder()

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = encoder.fit_transform(data[col])


    # ==========================================
    # SCALE FEATURES (VERY IMPORTANT)
    # ==========================================

    data_scaled = scaler.transform(data)


    # ==========================================
    # PREDICTION BUTTON
    # ==========================================

    if st.button("🚀 Detect Fraud"):

        predictions = model.predict(data_scaled)
        prob = model.predict_proba(data_scaled)[:, 1]

        result_df = data.copy()
        result_df["Fraud_Prediction"] = predictions
        result_df["Fraud_Probability"] = prob

        st.subheader("✅ Prediction Results")
        st.dataframe(result_df)

        fraud_count = np.sum(predictions)
        total = len(predictions)

        st.success(f"Fraud Transactions Detected: {fraud_count} / {total}")
