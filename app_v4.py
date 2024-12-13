import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# Configure the Gemini API using the environment variable
genai.configure(api_key=os.environ("GEMINI_API_KEY"))

# Load the new trained models
rf_fertility = load("rf_fertility_model.joblib")
rf_regular_cycle = load("rf_regular_cycle_model.joblib")

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Cycle Prediction and Feedback with Gemini",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the app with custom styling
st.markdown("""
    <h1 style="text-align:center; color:#4C8C91;">Cycle Prediction and Feedback with Gemini</h1>
    <p style="text-align:center; font-size:18px; color:#555;">Track and predict your menstrual cycle with advanced AI.</p>
""", unsafe_allow_html=True)

# Privacy Notice
st.markdown("""
    <div style="background-color:#F1D0D6; padding:15px; text-align:center; color:#333; font-size:16px;">
        <strong>Privacy and Security Notice:</strong><br>
        We value your privacy. All data entered is processed securely and is not shared with any third parties. 
        Your personal information will never be sold or shared. Your data is used solely to provide personalized cycle predictions and support.
    </div>
""", unsafe_allow_html=True)

# User Inputs
st.write("### Please enter your cycle details:")
cycle_length = st.number_input("Cycle Length (Days)", min_value=1, value=28, help="Enter the length of your cycle in days.")
average_cycle_length = st.number_input("Average Cycle Length (Days)", min_value=1, value=28, help="Enter the average cycle length.")
ovulation_day = st.number_input("Estimated Ovulation Day", min_value=1, max_value=31, value=14, help="Select the estimated ovulation day.")
luteal_phase_length = st.number_input("Luteal Phase Length (Days)", min_value=1, max_value=18, value=12, help="Enter the luteal phase length.")
high_fertility_start = st.number_input("High Fertility Start (Days)", min_value=1, max_value=31, value=12, help="Enter the start of high fertility days.")
peak_cycle = st.selectbox("Peak Cycle (Yes=1, No=0)", [1, 0])
body_mass_index = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
reproductive_status = st.selectbox("Reproductive Status (Fertile=1, Not Fertile=0)", [1, 0])

# Prediction Input Data
input_data = pd.DataFrame({
    "Cycle Length": [cycle_length],
    "Average Cycle Length": [average_cycle_length],
    "Ovulation Day": [ovulation_day],
    "Luteal Phase Length": [luteal_phase_length],
    "High Fertility Start": [high_fertility_start],
    "Peak Cycle": [peak_cycle],
    "Body Mass Index": [body_mass_index],
    "Reproductive Status": [reproductive_status]
})

# Predict Fertility
fertility_status = rf_fertility.predict(input_data)[0]
fertility_message = "High Fertility" if fertility_status == 1 else "Low Fertility"
fertility_color = "#2A9D8F" if fertility_status == 1 else "#A8DADC"

st.markdown(f"""
    <div style="background-color:{fertility_color}; padding:10px; text-align:center; color:white; font-size:22px; font-weight:bold;">
        Fertility Status: {fertility_message}
    </div>
""", unsafe_allow_html=True)

# Circular Visualization for Fertility
fig, ax = plt.subplots(figsize=(3, 3))
circle = plt.Circle((0.5, 0.5), 0.4, color=fertility_color, ec="black", lw=3)
ax.add_artist(circle)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
st.pyplot(fig)

# Predict Cycle Regularity
if st.button('🔮 Predict Cycle Regularity', key="predict_button"):
    cycle_regular_status = rf_regular_cycle.predict(input_data)[0]
    irregularity_message = "Irregular Cycle" if cycle_regular_status == 1 else "Regular Cycle"
    irregularity_color = "#457B9D" if cycle_regular_status == 1 else "#2A9D8F"
    
    st.markdown(f"""
        <div style="background-color:{irregularity_color}; padding:10px; text-align:center; color:white; font-size:22px; font-weight:bold;">
            {irregularity_message}
        </div>
    """, unsafe_allow_html=True)

# Gemini API Integration
st.write("### Ask Gemini about your cycle:")
user_question = st.text_input("Ask a question about your cycle", help="Type your question here.")
if user_question:
    try:
        gemini_response = genai.generate_text(prompt=user_question)
        st.success(f"Gemini's Response: {gemini_response['text']}")
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")

# Footer
st.markdown("""
    <footer style="text-align:center; padding: 20px; font-size: 14px; background-color:#A8DADC; color:white;">
        Powered by Streamlit & Gemini AI. Created with ❤️ for women's health.
    </footer>
""", unsafe_allow_html=True)

