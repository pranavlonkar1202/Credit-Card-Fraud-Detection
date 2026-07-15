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