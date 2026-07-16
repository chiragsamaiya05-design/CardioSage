from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "heart.csv"



MODEL_PATH = APP_DIR / "models" / "model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.5:
        return "Higher predicted risk", "risk-high"
    return "Lower predicted risk", "risk-low"


st.set_page_config(page_title="CardioCheck", page_icon="+", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(135deg, #f6f2ea 0%, #e8f1ed 48%, #dce7e1 100%); }
      h1, h2, h3 { color: #143d35; font-family: Georgia, serif; }
      .hero { padding: 2.6rem 0 1.4rem; }
      .eyebrow { color: #b43b38; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.13em; }
      .subtitle { color: #42635a; font-size: 1.08rem; max-width: 42rem; }
      .result { border-radius: 16px; padding: 1.4rem 1.6rem; margin: 1rem 0; }
      .risk-high { background: #f9dfda; border-left: 6px solid #b43b38; color: #63201f; }
      .risk-low { background: #dbeee4; border-left: 6px solid #287158; color: #123e30; }
      .note { color: #5c6e68; font-size: 0.88rem; }
      div.stButton > button { background: #143d35; color: #fff; border: 0; border-radius: 8px; padding: 0.65rem 1.6rem; }
      div.stButton > button:hover { background: #1d584c; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">CARDIOVASCULAR RISK SCREENING</div>
      <h1>CardioCheck</h1>
      <p class="subtitle">Enter the available clinical measurements to receive a model-based heart-disease risk estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Patient profile")
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("Sex", ["M", "F"], format_func=lambda value: "Male" if value == "M" else "Female")
        chest_pain = st.selectbox(
            "Chest pain type",
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA: atypical angina, NAP: non-anginal pain, TA: typical angina, ASY: asymptomatic.",
        )
        resting_bp = st.number_input("Resting blood pressure (mm Hg)", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=700, value=200)
        fasting_bs = st.radio("Fasting blood sugar above 120 mg/dL?", [0, 1], horizontal=True, format_func=lambda value: "Yes" if value else "No")
    with right:
        st.subheader("ECG and exercise")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Maximum heart rate", min_value=40, max_value=250, value=150)
        exercise_angina = st.selectbox("Exercise-induced angina", ["N", "Y"], format_func=lambda value: "No" if value == "N" else "Yes")
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Estimate risk")

if submitted:
    input_data = pd.DataFrame(
        [{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }]
    )
    probability = float(train_model().predict_proba(input_data)[0][1])
    label, style = risk_label(probability)
    st.markdown(
        f"<div class='result {style}'><h3>{label}</h3><p>Estimated probability: <strong>{probability:.0%}</strong></p></div>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    "<p class='note'>Educational demonstration only. This tool is not a medical diagnosis and must not replace advice from a qualified healthcare professional.</p>",
    unsafe_allow_html=True,
)
