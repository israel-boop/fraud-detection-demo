import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('sherlock_fraud_model.pkl')
    scaler = joblib.load('sherlock_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Define feature names (must match training)
# The notebook created these columns: Time, V1..V28, Amount, Hour, transactions_last_hour, amount_ratio
# We'll create input fields for the most important ones, and set others to typical values.
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
                 'Hour', 'transactions_last_hour', 'amount_ratio']

# We'll use typical values from the dataset for most features (mean of training set)
# But you can load actual means from a file, or hardcode.
# For simplicity, we'll set a default dict.
default_values = {
    'Time': 0,
    'V1': 0.0, 'V2': 0.0, 'V3': 0.0, 'V4': 0.0, 'V5': 0.0, 'V6': 0.0, 'V7': 0.0,
    'V8': 0.0, 'V9': 0.0, 'V10': 0.0, 'V11': 0.0, 'V12': 0.0, 'V13': 0.0, 'V14': 0.0,
    'V15': 0.0, 'V16': 0.0, 'V17': 0.0, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0,
    'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0, 'V26': 0.0,
    'V27': 0.0, 'V28': 0.0,
    'Amount': 50.0,
    'Hour': 12,
    'transactions_last_hour': 10,
    'amount_ratio': 1.0
}

st.set_page_config(page_title="Sherlock Fraud Detector", layout="centered")
st.title("🕵️ Sherlock Fraud Detector")
st.markdown("Enter transaction details to see if it's suspicious.")

# Create input fields for the most important features (V14, V10, Amount, Hour)
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=0.1)
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
with col2:
    v14 = st.number_input("V14 (anonymized feature)", value=-0.5, step=0.1)
    v10 = st.number_input("V10 (anonymized feature)", value=0.2, step=0.1)

# For simplicity, we'll keep other features at default values.
# In a production app, you might allow all features, but for demo this is fine.
input_dict = default_values.copy()
input_dict['Amount'] = amount
input_dict['Hour'] = hour
input_dict['V14'] = v14
input_dict['V10'] = v10

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale the required columns (same as during training)
cols_to_scale = ['Time', 'Amount', 'transactions_last_hour', 'amount_ratio']
input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

# Predict
if st.button("Investigate Transaction", type="primary"):
    proba = model.predict_proba(input_df)[0, 1]
    # Use threshold from notebook (you can hardcode the optimal threshold found, e.g., 0.3)
    threshold = 0.3  # Replace with your optimal threshold from the notebook
    pred = (proba >= threshold).astype(int)
    
    st.subheader("Result")
    if pred == 1:
        st.error(f"🚨 **FRAUD ALERT!** (Probability: {proba:.4f})")
    else:
        st.success(f"✅ Transaction appears normal. (Probability: {proba:.4f})")
    
    # Show probability gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = proba * 100,
        title = {'text': "Fraud Probability (%)"},
        domain = {'x': [0,1], 'y': [0,1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if proba > threshold else "darkgreen"},
            'steps': [
                {'range': [0, threshold*100], 'color': "lightgreen"},
                {'range': [threshold*100, 100], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold*100}}))
    st.plotly_chart(fig)
    
    # Simple explanation based on feature values
    st.subheader("Investigation Notes")
    notes = []
    if amount < 50:
        notes.append("• Small transaction amount (common fraud pattern).")
    if hour < 5 or hour > 22:
        notes.append("• Unusual hour of day (late night/early morning).")
    if v14 < -2 or v14 > 2:
        notes.append(f"• Extreme V14 value ({v14:.2f}) – a key fraud indicator.")
    if v10 < -2 or v10 > 2:
        notes.append(f"• Extreme V10 value ({v10:.2f}) – another fraud correlate.")
    if notes:
        st.markdown("\n".join(notes))
    else:
        st.markdown("No obvious red flags from basic checks.")