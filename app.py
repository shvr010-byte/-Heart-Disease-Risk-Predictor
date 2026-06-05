

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Card-style sections */
.risk-card {
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    font-family: 'DM Sans', sans-serif;
}
.high-risk {
    background: linear-gradient(135deg, #ffe4e4, #ffd0d0);
    border-left: 5px solid #e53e3e;
    color: #742a2a;
}
.low-risk {
    background: linear-gradient(135deg, #e6ffec, #d0f5dc);
    border-left: 5px solid #38a169;
    color: #1a4731;
}
.factor-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    margin: 4px 4px 4px 0;
    font-weight: 600;
}
.pill-red   { background:#fed7d7; color:#c53030; }
.pill-green { background:#c6f6d5; color:#276749; }
.pill-yellow{ background:#fefcbf; color:#744210; }
.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #718096;
    margin: 20px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = joblib.load("KNN_heart.pkl")
    scaler   = joblib.load("heart_scaler.pkl")
    columns  = joblib.load("heart_columns.pkl")
    return model, scaler, columns

try:
    model, scaler, expected_columns = load_artefacts()
    model_loaded = True
except Exception as e:
    st.warning(f"⚠️ Model files not found ({e}). Running in demo mode.")
    model_loaded = False


# ── Risk explanation logic ────────────────────────────────────────────────────
def explain_risk(age, sex, chest_pain, resting_bp, cholesterol,
                 fasting_bs, resting_ecg, max_hr, exercise_angina,
                 oldpeak, st_slope, prediction):
    """
    Returns (risk_factors, protective_factors, risk_score_10) based on
    medically grounded heuristics that mirror the model features.
    """
    risk_factors       = []
    protective_factors = []
    score              = 0   # accumulate raw points → scale to /10

    # Age
    if age >= 65:
        risk_factors.append(("Age ≥ 65", f"Your age ({age}) significantly raises cardiac risk."))
        score += 2
    elif age >= 50:
        risk_factors.append(("Age 50–64", f"Middle-to-older age ({age}) is a moderate risk factor."))
        score += 1
    else:
        protective_factors.append(("Younger Age", f"Age {age} is a protective factor for heart disease."))

    # Sex  (males have higher baseline risk in this dataset)
    if sex == "M":
        risk_factors.append(("Male Sex", "Males show higher heart disease prevalence in this dataset."))
        score += 1
    else:
        protective_factors.append(("Female Sex", "Females tend to have lower risk in this population."))

    # Chest Pain
    if chest_pain == "ASY":
        risk_factors.append(("Asymptomatic Chest Pain", "ASY (no chest pain) is paradoxically the highest-risk type — silent ischemia."))
        score += 2
    elif chest_pain == "TA":
        risk_factors.append(("Typical Angina", "Typical angina pain strongly suggests coronary artery disease."))
        score += 1.5
    elif chest_pain == "NAP":
        risk_factors.append(("Non-Anginal Pain", "Non-anginal pain carries moderate cardiac risk."))
        score += 0.5
    else:
        protective_factors.append(("Atypical Angina", "ATA (atypical angina) carries the lowest cardiac risk among chest pain types."))

    # Resting BP
    if resting_bp >= 140:
        risk_factors.append(("High Blood Pressure", f"Resting BP of {resting_bp} mmHg is Stage 2 hypertension — a major risk factor."))
        score += 1.5
    elif resting_bp >= 130:
        risk_factors.append(("Elevated BP", f"BP of {resting_bp} mmHg is elevated (Stage 1 hypertension)."))
        score += 0.8
    else:
        protective_factors.append(("Normal Blood Pressure", f"Your BP ({resting_bp} mmHg) is within a healthy range."))

    # Cholesterol
    if cholesterol >= 240:
        risk_factors.append(("High Cholesterol", f"{cholesterol} mg/dL is high and promotes arterial plaque."))
        score += 1.5
    elif cholesterol >= 200:
        risk_factors.append(("Borderline Cholesterol", f"{cholesterol} mg/dL is borderline high."))
        score += 0.5
    else:
        protective_factors.append(("Good Cholesterol Level", f"{cholesterol} mg/dL is within a desirable range."))

    # Fasting Blood Sugar
    if fasting_bs == 1:
        risk_factors.append(("High Fasting Blood Sugar", "FBS > 120 mg/dL suggests diabetes/pre-diabetes, a strong cardiac risk factor."))
        score += 1
    else:
        protective_factors.append(("Normal Fasting Blood Sugar", "FBS ≤ 120 mg/dL — no diabetic risk signal."))

    # Resting ECG
    if resting_ecg == "ST":
        risk_factors.append(("ST-T Abnormality on ECG", "ST-T wave changes indicate possible ischemia or electrolyte issues."))
        score += 1
    elif resting_ecg == "LVH":
        risk_factors.append(("Left Ventricular Hypertrophy", "LVH on ECG suggests long-standing high blood pressure or valve disease."))
        score += 1
    else:
        protective_factors.append(("Normal Resting ECG", "No ECG abnormality detected — reassuring sign."))

    # Max Heart Rate
    expected_max = 220 - age
    pct = (max_hr / expected_max) * 100
    if max_hr < 100:
        risk_factors.append(("Low Max Heart Rate", f"MaxHR of {max_hr} bpm is very low ({pct:.0f}% of expected {expected_max}). Low exercise capacity is a risk marker."))
        score += 1.5
    elif max_hr < 120:
        risk_factors.append(("Below-Average Max HR", f"MaxHR of {max_hr} bpm ({pct:.0f}% of expected) is below average."))
        score += 0.8
    else:
        protective_factors.append(("Good Aerobic Capacity", f"MaxHR of {max_hr} bpm ({pct:.0f}% of expected) suggests healthy cardiovascular fitness."))

    # Exercise-Induced Angina
    if exercise_angina == "Y":
        risk_factors.append(("Exercise-Induced Angina", "Chest pain triggered by exercise is a classic sign of obstructive coronary artery disease."))
        score += 2
    else:
        protective_factors.append(("No Exercise Angina", "No chest pain during exercise — a reassuring finding."))

    # Oldpeak
    if oldpeak >= 3.0:
        risk_factors.append(("High ST Depression (Oldpeak)", f"Oldpeak of {oldpeak} mm is a strong marker of myocardial ischemia."))
        score += 2
    elif oldpeak >= 1.5:
        risk_factors.append(("Moderate ST Depression", f"Oldpeak of {oldpeak} mm suggests mild-to-moderate ischemia."))
        score += 1
    elif oldpeak > 0:
        risk_factors.append(("Mild ST Depression", f"Oldpeak of {oldpeak} mm — minimal but worth monitoring."))
        score += 0.3
    else:
        protective_factors.append(("No ST Depression", "Oldpeak = 0 — no exercise-induced ST depression, a healthy sign."))

    # ST Slope
    if st_slope == "Flat":
        risk_factors.append(("Flat ST Slope", "A flat ST slope during exercise strongly correlates with ischemia."))
        score += 1.5
    elif st_slope == "Down":
        risk_factors.append(("Downsloping ST Slope", "Downsloping ST is the most ominous pattern — highest ischemic risk."))
        score += 2
    else:
        protective_factors.append(("Upsloping ST Slope", "Upsloping ST slope is normal and a protective sign."))

    # Scale score to /10  (max theoretical score ≈ 18)
    raw_max = 18.0
    rating  = round(min(score / raw_max * 10, 10), 1)

    # Override: if model says low-risk but heuristic says high, cap at 4
    if prediction == 0 and rating > 5:
        rating = 4.5
    # If model says high-risk but heuristic says low, floor at 5
    if prediction == 1 and rating < 4:
        rating = 5.0

    return risk_factors, protective_factors, rating


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Fill in your clinical details below and click **Predict** for an instant risk assessment with personalised explanations.")

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">👤 Personal Info</p>', unsafe_allow_html=True)
    age  = st.slider("Age", 18, 100, 45)
    sex  = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")

    st.markdown('<p class="section-header">🩺 Chest & BP</p>', unsafe_allow_html=True)
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "ATA – Atypical Angina",
            "NAP": "NAP – Non-Anginal Pain",
            "TA":  "TA  – Typical Angina",
            "ASY": "ASY – Asymptomatic",
        }[x],
    )
    resting_bp  = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    st.markdown('<p class="section-header">🔬 Lab & ECG</p>', unsafe_allow_html=True)
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes (> 120)" if x == 1 else "No (≤ 120)",
    )
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST – ST-T Wave Abnormality",
            "LVH": "LVH – Left Ventricular Hypertrophy",
        }[x],
    )

    st.markdown('<p class="section-header">🏃 Exercise Test</p>', unsafe_allow_html=True)
    max_hr          = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
    oldpeak         = st.slider("Oldpeak – ST Depression", 0.0, 6.0, 0.0, step=0.1)
    st_slope        = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        format_func=lambda x: f"{x}sloping" if x != "Up" else "Upsloping (normal)",
    )

st.divider()
predict_btn = st.button("🔍 Predict My Risk", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analysing your data…"):

        if model_loaded:
            raw_input = {
                'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
                'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
                'Sex_' + sex: 1,
                'ChestPainType_' + chest_pain: 1,
                'RestingECG_' + resting_ecg: 1,
                'ExerciseAngina_' + exercise_angina: 1,
                'ST_Slope_' + st_slope: 1,
            }
            input_df = pd.DataFrame([raw_input])
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df     = input_df[expected_columns]
            scaled_input = scaler.transform(input_df)
            prediction   = model.predict(scaled_input)[0]
        else:
            # Demo fallback — simple rule for demo
            prediction = 1 if (exercise_angina == "Y" or oldpeak >= 2.0 or chest_pain == "ASY") else 0

        risk_factors, protective_factors, rating = explain_risk(
            age, sex, chest_pain, resting_bp, cholesterol,
            fasting_bs, resting_ecg, max_hr, exercise_angina,
            oldpeak, st_slope, prediction,
        )

    # ── Result card ──────────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown(
            '<div class="risk-card high-risk">'
            '<h2 style="margin:0 0 6px 0;">⚠️ High Risk of Heart Disease</h2>'
            '<p style="margin:0;">The model predicts a <strong>high likelihood</strong> of heart disease based on your inputs. '
            'Please consult a cardiologist for a thorough evaluation.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="risk-card low-risk">'
            '<h2 style="margin:0 0 6px 0;">✅ Low Risk of Heart Disease</h2>'
            '<p style="margin:0;">The model predicts a <strong>low likelihood</strong> of heart disease. '
            'Maintain a healthy lifestyle and continue regular check-ups.</p>'
            '</div>',
            unsafe_allow_html=True,
        )



    # ── Why is your risk high / low ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔎 Why is your risk like this?")

    if risk_factors:
        st.markdown("**🚨 Risk-Raising Factors:**")
        pills_html = ""
        for title, explanation in risk_factors:
            pills_html += f'<span class="factor-pill pill-red">⬆ {title}</span>'
        st.markdown(pills_html, unsafe_allow_html=True)



    # ── Advice ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if prediction == 1:
        st.markdown("### 💡 What You Can Do")
        st.markdown("""
- 🩺 **See a cardiologist** for a stress test, ECG, and lipid panel.
- 🥗 **Diet**: Reduce saturated fats and sodium; increase fibre and omega-3s.
- 🚶 **Exercise**: Aim for 150 min/week of moderate aerobic activity.
- 🚭 **Stop smoking** if applicable — it multiplies cardiac risk.
- 💊 **Medications**: Ask your doctor about statins / antihypertensives if BP or cholesterol are high.
- 🩸 **Manage blood sugar** if FBS is elevated.
        """)
    else:
        st.markdown("### 💡 Keep It Up!")
        st.markdown("""
- ✅ Maintain a heart-healthy diet rich in vegetables, whole grains, and lean proteins.
- 🏃 Keep up regular physical activity.
- 🔁 Schedule annual check-ups to monitor BP, cholesterol, and blood sugar.
- 😴 Prioritise 7–9 hours of quality sleep per night.
        """)

    st.info(" This tool is for educational purposes only and is not a substitute for professional medical advice.", icon="ℹ️")