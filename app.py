import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
try:
    model = joblib.load("credit_card_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection System")

st.write("""
Upload a CSV file containing credit card transactions.

The system will predict whether each transaction is:

- ✅ Normal
- 🚨 Fraud
""")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# -----------------------------
# If file uploaded
# -----------------------------
if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(df)

        # -----------------------------
        # Validate Dataset
        # -----------------------------
        expected_columns = [
            "Time","V1","V2","V3","V4","V5","V6","V7",
            "V8","V9","V10","V11","V12","V13","V14",
            "V15","V16","V17","V18","V19","V20",
            "V21","V22","V23","V24","V25","V26",
            "V27","V28","Amount"
        ]

        missing_columns = [
            col for col in expected_columns
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing columns: {missing_columns}"
            )
            st.stop()

        # -----------------------------
        # Prepare Features
        # -----------------------------
        if "Class" in df.columns:
            X = df.drop("Class", axis=1)
        else:
            X = df.copy()

        # -----------------------------
        # Scale Data
        # -----------------------------
        X_scaled = scaler.transform(X)

        # -----------------------------
        # Predict
        # -----------------------------
        with st.spinner("Predicting..."):

            prediction = model.predict(X_scaled)

        # Convert Prediction
        prediction_label = [
            "Normal" if p == 0 else "Fraud"
            for p in prediction
        ]

        df["Prediction"] = prediction_label

        # -----------------------------
        # Probability (if available)
        # -----------------------------
        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(X_scaled)

            df["Fraud Probability"] = probability[:, 1]

        # -----------------------------
        # Summary
        # -----------------------------
        normal = (df["Prediction"] == "Normal").sum()
        fraud = (df["Prediction"] == "Fraud").sum()

        total = len(df)

        st.subheader("Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total)

        col2.metric("Normal", normal)

        col3.metric("Fraud", fraud)

        # -----------------------------
        # Prediction Table
        # -----------------------------
        st.subheader("Prediction Results")

        st.dataframe(df)

        # -----------------------------
        # Download Button
        # -----------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        st.success("Prediction completed successfully!")

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info("Please upload a CSV file to start prediction.")