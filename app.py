import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

.block-container {
    padding-top: 2rem;
}

.header-box {
    background: linear-gradient(135deg, #1565C0, #42A5F5);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg,#1565C0,#42A5F5);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(135deg,#0D47A1,#1E88E5);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

# Cache resource ensures the model is only loaded ONCE, speeding up your app drastically
@st.cache_resource
def load_prediction_assets():
    model = load_model("titanic_ann_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_prediction_assets()
except Exception as e:
    st.error(f"Error loading model or scaler. Ensure 'titanic_ann_model.h5' and 'scaler.pkl' are in the repository. Details: {e}")
    st.stop()

# ---------------- HEADER ---------------- #

st.markdown("""
<div class="header-box">
<h1>🚢 Titanic Survival Prediction System</h1>
<h4>Deep Learning Based Passenger Survival Prediction</h4>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- LAYOUT ---------------- #

left_col, right_col = st.columns(2)

# ---------------- INPUT SECTION ---------------- #

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Passenger Details")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.slider("Age", 1, 80, 25)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
    
    predict = st.button("🔍 Predict Survival")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION SECTION ---------------- #

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Results")

    if predict:
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Age': [age],
            'Fare': [fare]
        })

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)
        probability = float(prediction[0][0])

        if probability > 0.5:
            result = "✅ Survived"
        else:
            result = "❌ Not Survived"

        confidence = max(probability, 1 - probability)

        # METRICS
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", result)
        c2.metric("Probability", f"{probability:.2f}")
        c3.metric("Confidence", f"{confidence:.2f}")

        st.write("")

        # SMALL DONUT CHART
        fig, ax = plt.subplots(figsize=(3, 3))
        values = [probability, 1 - probability]
        labels = ['Survive', 'Not Survive']
        colors = ['#42A5F5', '#ef5350']

        ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='w')
        )
        ax.axis('equal')
        
        st.pyplot(fig)
        plt.close(fig)  # Clean up memory

    else:
        st.info("Enter passenger details and click Predict Survival.")

    st.markdown('</div>', unsafe_allow_html=True)