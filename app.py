import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header Card */

.header-card {
    background: linear-gradient(135deg, #1565C0, #42A5F5);
    padding: 35px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}

/* Glassmorphism Cards */

.glass-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.3);
}

/* Prediction Result */

.result-success {
    background: linear-gradient(135deg,#43A047,#66BB6A);
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}

.result-fail {
    background: linear-gradient(135deg,#E53935,#EF5350);
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}

/* Button */

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg,#1565C0,#42A5F5);
    color: white;
    border: none;
    border-radius: 14px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg,#0D47A1,#1E88E5);
    color: white;
}

/* Metrics */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.75);
    border-radius: 18px;
    padding: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = tf.keras.models.load_model("titanic_ann_model.h5")

scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div class="header-card">
    <h1>🚢 Titanic Survival Prediction System</h1>
    <h4>Deep Learning Based Passenger Survival Prediction</h4>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------

left_col, right_col = st.columns([1, 1])

# ---------------------------------------------------
# INPUT PANEL
# ---------------------------------------------------

with left_col:

    st.markdown("""
    <div class="glass-card">
    <h2>🧾 Passenger Information</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        help="1 = First Class, 2 = Second Class, 3 = Third Class"
    )

    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=25
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0,
        step=1.0
    )

    st.write("")

    predict_btn = st.button("🔍 Predict Survival")

# ---------------------------------------------------
# OUTPUT PANEL
# ---------------------------------------------------

with right_col:

    st.markdown("""
    <div class="glass-card">
    <h2>📊 AI Prediction Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if predict_btn:

        # -------------------------------------------
        # INPUT DATA
        # -------------------------------------------

        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Age': [age],
            'Fare': [fare]
        })

        # -------------------------------------------
        # SCALING
        # -------------------------------------------

        input_scaled = scaler.transform(input_data)

        # -------------------------------------------
        # PREDICTION
        # -------------------------------------------

        prediction = model.predict(input_scaled, verbose=0)

        probability = float(prediction[0][0])

        survive_prob = probability
        nonsurvive_prob = 1 - probability

        confidence = max(survive_prob, nonsurvive_prob)

        # -------------------------------------------
        # RESULT
        # -------------------------------------------

        if survive_prob > 0.5:

            st.markdown(f"""
            <div class="result-success">
                ✅ SURVIVED
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-fail">
                ❌ NOT SURVIVED
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # -------------------------------------------
        # METRICS
        # -------------------------------------------

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Survival Probability",
                f"{survive_prob:.2f}"
            )

        with m2:
            st.metric(
                "Non-Survival",
                f"{nonsurvive_prob:.2f}"
            )

        with m3:
            st.metric(
                "Confidence",
                f"{confidence:.2f}"
            )

        st.write("")

        # -------------------------------------------
        # SMALL DONUT CHART
        # -------------------------------------------

        st.markdown("""
        <div class="glass-card">
        <h3 style='text-align:center;'>📈 Probability Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(2.8, 2.8))

        values = [survive_prob, nonsurvive_prob]
        labels = ['Survive', 'Not Survive']

        ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4)
        )

        ax.axis('equal')

        st.pyplot(fig)

    else:

        st.markdown("""
        <div class="glass-card">
        <h4 style='text-align:center; color:gray;'>
        Enter passenger details and click Predict Survival
        </h4>
        </div>
        """, unsafe_allow_html=True)