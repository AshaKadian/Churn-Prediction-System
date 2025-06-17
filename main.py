import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set Groq API credentials
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = os.getenv("GROQ_API_BASE")

# Load models
model = joblib.load(r"xgboost_model_final.pkl")  # Pipeline model

st.set_page_config(page_title="Caregiver Churn Prediction", layout="centered")
    st.title("🎯 Caregiver Churn Prediction System")
st.markdown("### 📝 Please Fill in the Caregiver Details:")

# --- Input Form ---
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("🧑 Gender", ["Female", "Male"], index=0)
    marital_status = st.selectbox("💍 Marital Status", ["Married", "Unknown", "Divorce", "Single", "Separated"])
    pay_unit = st.selectbox("💵 Pay Unit", ["Hourly", "Visit", "15 Min"], index=0)
    caregiver_can_do_nights = st.radio("🌙 Available for Night Shifts?", ["Yes", "No"], index=1)
    caregiver_can_do_days = st.radio("🌞 Available for Day Shifts?", ["Yes", "No"], index=0)

with col2:
    age = st.number_input("🎂 Age", min_value=18, step=1, value=30)
    caregiver_tenure_years = st.number_input("📆 Caregiver Tenure (Years)", min_value=0.0, step=0.1)
    pay_rate = st.number_input("💸 Pay Rate ($)", min_value=0.0, step=0.1)
    payroll_units_without_ot = st.number_input("⏱ Payroll Units Without OT", min_value=0.0, step=0.1)
    payroll_ot_amount = st.number_input("🕐 Payroll OT Amount", min_value=0.0, step=0.1)
    total_payroll_amount = st.number_input("💰 Total Payroll Amount", min_value=0.0, step=0.1)

# Service Unit sorted by frequency
service_unit_options = [
    "Personal Care", "HMAP INCL", "Palliative", "Live-In", "Respite Personal Care",
    "Post Operative Care", "HMAP CH Visit", "Caregiver Training", "RPNRN Shift", "Other",
    "Phone Consult", "Foot Care", "RPNRN Visit", "Delivery", "Supervisory Visit",
    "Covid 19 Rapid Antigen Testing", "PSW Visit", "PSW  Shift", "Couple Care",
    "Hospall HM Work", "On Call Duty", "Training"
]
service_unit = st.selectbox("🧾 Service Unit", service_unit_options, index=0)

# Race sorted by frequency
race_options = [
    "Middle Eastern Canadian", "Asian Canadian", "African Canadian", "Hispanic or Latino Canadian",
    "British Canadian", "Eastern European Canadian", "Caribbean Canadian", "Italian Canadian",
    "French Canadian", "Jewish Canadian"
]
race = st.selectbox("🌍 Race", race_options, index=0)

# Convert Yes/No to 1/0 for model input
caregiver_can_do_nights = 1 if caregiver_can_do_nights == "Yes" else 0
caregiver_can_do_days = 1 if caregiver_can_do_days == "Yes" else 0

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

    risk_label = "High Risk" if probability > 0.6 else "Moderate Risk" if probability > 0.4 else "Low Risk"

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

    if risk_label in ["High Risk", "Moderate Risk"]:
        # --- LLM-based Retention Strategy (Concise) ---
        st.subheader("📌 Suggested Retention Strategy (LLM)")
    
        try:
            prompt = f"""
            You are an HR strategist specializing in caregiver retention in home healthcare.

            A caregiver has a churn probability of {probability:.2f} and a value score of {value_score:.2f}. 
            They are {age} years old, earning ${pay_rate:.2f}/${pay_unit}, and have {caregiver_tenure_years:.1f} years of experience with us. 
            They {'can' if caregiver_can_do_nights else "cannot"} work night shifts and {'can' if caregiver_can_do_days else "cannot"} work day shifts.
    
            Create a **practical, time-phased retention strategy** that includes:
            Each bullet should have a **bolded subheading**, followed by a clear, actionable explanation. Every subheading bullet point should be in next line and not in the same line as the action detail.
            Format the strategy as follows:
            ##### 1. Immediate Actions (within 1 week)
            - **[Subheading]:** [Action detail]
            - **[Subheading]:** [Action detail]
            - **[Subheading]:** [Action detail]
    
            ##### 2. Short-Term Actions (within 1-2 months)
            - **[Subheading]:** [Action detail]
            - **[Subheading]:** [Action detail]
            - **[Subheading]:** [Action detail]
    
            ##### 3. Long-Term Actions (over 3-6 months)
            - **[Subheading]:** [Action detail]
            - **[Subheading]:** [Action detail]
            - **[Subheading]:** [Action detail]
            
            Format clearly using pointers and section titles. Be concise, action-driven, and motivational. Strategy must be respectful, cost-aware, and relevant to the profile above. Do not include any disclaimers or unnecessary narrative — just the structured retention strategy.
    
            Length: Max 70 words. Output only the formatted strategy.
            """
    
            response = openai.ChatCompletion.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            concise_strategy = response.choices[0].message.content.strip()
            st.markdown(f"{concise_strategy}", unsafe_allow_html=True)
    
        except Exception as e:
            st.warning(f"LLM could not generate strategy. Error: {e}")
