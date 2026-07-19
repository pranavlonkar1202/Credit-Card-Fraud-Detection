import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_card_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection System")

st.write("""
Welcome!

This application predicts whether a credit card transaction is **Normal** or **Fraudulent** using a trained Machine Learning model.

Upload a CSV file to begin.
""")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df)

expected_columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7",
    "V8","V9","V10","V11","V12","V13","V14",
    "V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26",
    "V27","V28","Amount"
]

missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    st.error(f"The uploaded CSV is missing these columns: {missing_columns}")
    st.stop()

if "Class" in df.columns:
    X = df.drop("Class", axis=1)
else:
    X = df.copy()

X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)

prediction_label = []

for value in prediction:
    if value == 0:
        prediction_label.append("Normal")
    else:
        prediction_label.append("Fraud")

df["Prediction"] = prediction_label

st.subheader("Prediction Results")

st.dataframe(df)

probability = model.predict_proba(X_scaled)

df["Fraud Probability"] = probability[:,1]

csv = df.to_csv(index=False)

st.download_button(
    label="Download Results",
    data=csv,
    file_name="prediction_results.csv",
    mime="text/csv"
)