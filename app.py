"""
Diabetes Risk Prediction — Streamlit App (starter)

This is a placeholder app. Once the final model is saved (see notebook 03),
load it here and build out the input form + prediction display.

Run with: streamlit run app/app.py
"""

import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("Diabetes Risk Predictor")
st.write(
    "Estimate elevated risk of prediabetes/diabetes based on health indicators. "
    "This tool is for educational purposes only and is not a medical diagnosis."
)

st.info(
    "🚧 This app is under construction. Once the final model is trained and saved "
    "(see notebooks/03_model_comparison_and_evaluation.ipynb), this page will let you "
    "enter health indicators and get a live risk prediction with a SHAP-based explanation."
)

# --- Planned input form (uncomment and wire up once the model is ready) ---
# with st.form("risk_form"):
#     bmi = st.slider("BMI", 12.0, 60.0, 25.0)
#     age_group = st.selectbox("Age group", options=[...])
#     high_bp = st.checkbox("High blood pressure")
#     high_chol = st.checkbox("High cholesterol")
#     gen_health = st.select_slider("General health (1=excellent, 5=poor)", options=[1, 2, 3, 4, 5])
#     submitted = st.form_submit_button("Predict risk")
#
#     if submitted:
#         # model = joblib.load("src/final_model.pkl")
#         # prediction = model.predict_proba([[...]])[:, 1]
#         # st.metric("Predicted risk", f"{prediction[0]:.1%}")
#         pass
