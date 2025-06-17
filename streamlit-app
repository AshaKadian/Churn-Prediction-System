import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page setup
st.set_page_config(page_title="Caregiver Churn Prediction", layout="centered")

# Title and subtitle
st.title("📊 Caregiver Churn Prediction")
st.markdown("Use the form below to predict the likelihood of a caregiver leaving.")

# Load trained model and feature columns
try:
    model = joblib.load("xgboost_model1.pkl")

    # Check if model has attribute feature_names_in_
    if hasattr(model, "feature_names_in_"):
        expected_cols = [col for col in model.feature_names_in_ if col != "Prediction"]
    elif hasattr(model, "named_steps"):  # If it's a pipeline
        expected_cols = [col for col in model.named_steps['model'].feature_names_in_ if col != "Prediction"]
    else:
        raise AttributeError("Cannot find feature names in the loaded model.")

    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# --- User input form ---
def user_input_features():
    st.header("📁 Employment Details")
    col1, col2 = st.columns(2)
    with col1:
        service_unit = st.selectbox("Service Unit", [
            "Personal Care", "HMAP INCL", "Palliative", "Live-In", "Respite Personal Care",
            "Post Operative Care", "HMAP CH Visit", "Caregiver Training", "RPNRN Shift", "Other",
            "Phone Consult", "Foot Care", "RPNRN Visit", "Delivery", "Supervisory Visit",
            "Covid 19 Rapid Antigen Testing", "PSW Visit", "PSW  Shift", "Couple Care",
            "Hospall HM Work", "On Call Duty", "Training"
        ])
        pay_unit = st.selectbox("Pay Unit", ["Hourly", "Visit", "15 Min"])
    with col2:
        pay_rate = st.number_input("Pay Rate", 0.0, 1000.0, 100.0)
        payroll_units_without_ot = st.number_input("Payroll UnitsWithoutOT", 0.0, 100.0, 40.0)
        payroll_ot_amount = st.number_input("Payroll OTAmount", 0.0, 100.0, 5.0)
        total_payroll_amount = st.number_input("Total Payroll Amount", 0.0, 10000.0, 1200.0)

    st.header("🧍 Personal Details")
    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 70, 30)
        marital_status = st.selectbox("Marital Status", ["Married", "Unknown", "Divorce", "Single", "Separated"])
    with col4:
        race = st.selectbox("Race", [
            "Middle Eastern Canadian", "Asian Canadian", "African Canadian", "Hispanic or Latino Canadian",
            "British Canadian", "Eastern European Canadian", "Caribbean Canadian", "Italian Canadian",
            "French Canadian", "Jewish Canadian"
        ])

    st.header("⏰ Shift Preferences")
    col5, col6 = st.columns(2)
    with col5:
        can_do_nights = st.selectbox("Can do Night Shifts?", ["No", "Yes"])
    with col6:
        can_do_days = st.selectbox("Can do Day Shifts?", ["No", "Yes"])

    # Create dictionary
    data = {
        "Service Unit": service_unit,
        "Pay Unit": pay_unit,
        "Pay Rate": pay_rate,
        "Payroll UnitsWithoutOT": payroll_units_without_ot,
        "Payroll OTAmount": payroll_ot_amount,
        "Total Payroll Amount": total_payroll_amount,
        "Gender": gender,
        "Age": age,
        "Marital Status": marital_status,
        "Race": race,
        "Caregiver Attributes_Can do Nights": 1 if can_do_nights == "Yes" else 0,
        "Caregiver Attributes_Can do Days": 1 if can_do_days == "Yes" else 0
    }

    df = pd.DataFrame([data])

    # Add missing expected columns with default values
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match model input
    df = df[expected_cols]
    return df

# --- Main app logic ---
input_df = user_input_features()

# Temporary workaround (not ideal):
if 'Prediction' in model.feature_names_in_:
    input_df['Prediction'] = 0  # Dummy column just to satisfy model

# Reorder columns
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)


# Prediction
if st.button("🔍 Predict Churn"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        st.subheader("🔔 Prediction Result")
        if prediction == 1:
            st.error(f"🚨 High Risk: Caregiver likely to churn ({probability * 100:.2f}%)")
        else:
            st.success(f"✅ Low Risk: Caregiver likely to stay ({(1 - probability) * 100:.2f}%)")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Show user input for verification
if st.checkbox("📋 Show Input Data"):
    st.write(input_df)
