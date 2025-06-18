# Caregiver Churn Prediction and Retention Strategy

A Streamlit-based interactive application to **predict caregiver churn** and generate **personalized retention strategies** using a trained XGBoost model and LLM (via Groq API).

---

## Features

- **Churn Prediction**: Predict whether a caregiver is likely to churn based on their attributes.
- **Churn Risk Labeling**: Categorizes caregivers into `Low`, `Moderate`, or `High Risk`.
- **Employee Value Score**: Quantifies caregiver value using tenure, pay rate, and payroll.
- **Explainable AI**: Visualize key feature impact using SHAP values.
- **LLM-Powered Retention Plans**: For high and moderate churn risks, a structured plan is generated using Groq’s LLaMA-3.
- **User-Friendly Interface**: Streamlit app with emojis, sectioning, and two-column layout for better usability.

---

## Project Structure

main/
│
├── Churn_EDA.ipynb                # Exploratory data analysis
├── Churn_preprocessing.ipynb      # Data cleaning & preprocessing
├── Churn_training.ipynb           # Model training and evaluation
│
├── cleaned_dataset.csv            # Cleaned version of the dataset
├── dataset_188.csv                # Original or raw dataset
│
├── hyperparameter_tuning/         # Folder for tuning experiments
│
├── main.py                        # Streamlit app for prediction & strategy generation
├── xgboost_model_final.pkl        # Trained XGBoost model pipeline (preprocessing + model)
├── requirements.txt               # Python dependencies
│
├── streamlit-app/                 # Extra Streamlit file for prediction

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/caregiver-churn-app.git
cd caregiver-churn-app

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install dependencies

```bash
pip install -r requirements.txt

### 4. Add your Groq API credentials to a `.env` file

Create a `.env` file in the root directory and add the following lines:

```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_API_BASE=https://api.groq.com/openai/v1

### 5. Run the Streamlit app

```bash
streamlit run main.py

---

## Value Score Calculation

A custom **Employee Value Score** is calculated using the following formula:

```python
value_score = (pay_rate * 0.2) + (tenure_years * 0.5) + (total_payroll_amount * 0.3)

### Value Labels

- **High Value**: Score > 80  
- **Moderate Value**: 40 < Score ≤ 80  
- **Low Value**: Score ≤ 40

---

## LLM Retention Strategy Logic

A custom prompt is sent to **LLaMA-3** (via **Groq API**) including:

- Churn probability  
- Value score  
- Tenure, shift availability, and pay rate  

### LLM Response Format: Retention Strategy

The generated retention strategy follows a structured, time-phased format to ensure clarity and impact:

#### 1. Immediate Actions (Within 1 Week)
- **[Action Title]:** Clear and specific action to be taken immediately.

#### 2. Short-Term Actions (Within 1–2 Months)
- **[Action Title]:** Mid-term initiatives focused on engagement, development, or incentives.

#### 3. Long-Term Actions (Over 3–6 Months)
- **[Action Title]:** Sustained strategies for career growth, wellness, and retention monitoring.

Each action includes a **bolded heading** and a concise description. The output is designed to be practical, motivational, and budget-conscious.

---

## Security Note

**Do NOT hardcode API keys in `main.py`.**

❗ Use the `.env` file and `python-dotenv` to load credentials securely:

```python
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")

---

## Key Packages & Dependencies

- **`streamlit`** – Web-based interactive UI  
- **`joblib`** – Model serialization and loading  
- **`pandas`**, **`matplotlib`**, **`shap`** – Data manipulation, visualization, and model explainability  
- **`scikit-learn`**, **`xgboost`** – Preprocessing, model training, and classification  
- **`openai`** – LLM integration using Groq's API for personalized retention strategies  
- **`python-dotenv`** – Securely loads environment variables like API keys from a `.env` file 

---

## Future Improvements

- Implement dynamic normalization for the value score (avoid hardcoding max values)  
- Add history logging for predictions and retention strategies  
- Deploy to **Streamlit Cloud** or **AWS**  
- Develop an **admin dashboard** with filters, export options, and activity logs  

---

## Authors

**Khushleen**,  **Asha**, **Astika**
