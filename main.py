
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import openai
import os

# Set Groq API key
openai.api_key = "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"
openai.api_base = "https://api.groq.com/openai/v1"

# Load models
model = joblib.load(r"C:\Users\Nisha kadian\Downloads\xgboost_model_final.pkl")  # Pipeline model


st.set_page_config(page_title="Caregiver Churn Prediction", layout="centered")
st.title("Caregiver Churn & Retention Strategy")
st.markdown("### Please fill in the caregiver details:")

# --- Input Form ---
gender = st.selectbox("Gender", ["Female", "Male"])
race = st.selectbox("Race", [
    "Middle Eastern Canadian", "Hispanic or Latino Canadian", "Asian Canadian",
    "Eastern European Canadian", "Hispanic Canadian", "African Canadian",
    "British Canadian", "South Asian Canadian"
])
marital_status = st.selectbox("Marital Status", ["Unknown", "Married", "Single", "Divorce"])
service_unit = st.selectbox("Service Unit", [
    "Personal Care", "HMAP INCL", "Palliative", "Foot Care", "Caregiver Training",
    "Post Operative Care", "Respite Personal Care", "RPNRN Shift"
])
pay_unit = st.selectbox("Pay Unit", ["Hourly", "Visit"])
pay_rate = st.number_input("Pay Rate", min_value=0.0, step=0.1)
payroll_units_without_ot = st.number_input("Payroll Units Without OT", min_value=0.0, step=0.1)
payroll_ot_amount = st.number_input("Payroll OT Amount", min_value=0.0, step=0.1)
total_payroll_amount = st.number_input("Total Payroll Amount", min_value=0.0, step=0.1)
age = st.number_input("Age", min_value=0, step=1)
caregiver_can_do_nights = st.selectbox("Can do Nights (1 = Yes, 0 = No)", [1, 0])
caregiver_can_do_days = st.selectbox("Can do Days (1 = Yes, 0 = No)", [1, 0])
caregiver_tenure_years = st.number_input("Caregiver Tenure (Years)", min_value=0.0, step=0.1)

# --- Prepare input ---
input_data = pd.DataFrame([{
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
    "Caregiver Attributes_Can do Nights": caregiver_can_do_nights,
    "Caregiver Attributes_Can do Days": caregiver_can_do_days,
    "CaregiverTenureYears": caregiver_tenure_years
}])

if st.button("Predict Churn & Strategy"):
    # --- Churn Prediction ---
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_label = "High Risk" if probability > 0.6 else "Moderate Risk" if probability > 0.3 else "Low Risk"

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ {risk_label}: The caregiver is likely to churn ({probability * 100:.2f}%)")
    else:
        st.success(f"✅ {risk_label}: The caregiver is likely to stay ({(1 - probability) * 100:.2f}%)")

    # --- Value Score ---
    value_score = (pay_rate * 0.2 + caregiver_tenure_years * 0.5 + total_payroll_amount * 0.3)
    value_label = "High Value" if value_score > 80 else "Moderate Value" if value_score > 40 else "Low Value"
    st.markdown(f"**💎 Employee Value Score:** {value_score:.2f} ({value_label})")

    # --- SHAP Feature Importance ---
    st.subheader("🔍 Feature Impact (Top Predictors)")
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed_data = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.Explainer(classifier)
    shap_values = explainer(transformed_data)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap_df.plot(kind="barh", x="Feature", y="SHAP Value", ax=ax, color="skyblue", legend=False)
    ax.invert_yaxis()
    ax.set_title("Top Contributing Features")
    st.pyplot(fig)

    with st.expander("🔬 See SHAP Value Table"):
        st.dataframe(shap_df)

    # --- LLM-based Retention Strategy (Concise) ---
    st.subheader("📌 Suggested Retention Strategy (LLM)")

    try:
        prompt = f"""
        Based on the following caregiver attributes:
        - Churn Probability: {probability:.2f}
        - Value Score: {value_score:.2f}
        - Age: {age}, Pay Rate: {pay_rate}, Tenure: {caregiver_tenure_years} years

        Please provide a short and effective 1-paragraph retention strategy to reduce the chance of churn. Keep it practical and avoid long descriptions.
        """

        response = openai.ChatCompletion.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        concise_strategy = response.choices[0].message.content.strip()
        st.markdown(f"**{concise_strategy}**")

    except Exception as e:
        st.warning(f"LLM could not generate strategy. Error: {e}")
