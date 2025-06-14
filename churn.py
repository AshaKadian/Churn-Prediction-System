import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# Load model
model = joblib.load(r"C:\Users\Nisha kadian\Downloads\xgboost_model_final.pkl")

st.set_page_config(page_title="Caregiver Churn Prediction", layout="centered")
st.title("Caregiver Churn Prediction")
st.markdown("### Please fill in the caregiver details:")

# Input
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

# Prepare input
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

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk: The caregiver is likely to churn ({probability * 100:.2f}%)")
    else:
        st.success(f"✅ Low Risk: The caregiver is likely to stay ({(1 - probability) * 100:.2f}%)")

    st.subheader("🔍 Feature Impact (Top Predictors)")

    # SHAP Explanation
    # Access preprocessor and classifier separately
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform input data
    transformed_data = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.Explainer(classifier)
    shap_values = explainer(transformed_data)

    # Get top N contributing features
    N = 10
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False).head(N)

    # Plot as bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    shap_df.plot(kind="barh", x="Feature", y="SHAP Value", ax=ax, color="skyblue", legend=False)
    ax.invert_yaxis()
    ax.set_title("Top Contributing Features")
    st.pyplot(fig)

    # Optional: Raw SHAP table
    with st.expander("See Top SHAP Value Table"):
        st.dataframe(shap_df)

