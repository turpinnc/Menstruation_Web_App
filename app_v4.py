import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables (if needed)
load_dotenv()

# Configure the Gemini API using the environment variable
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load the models from pickle files
with open('rf_fertility_model.pkl', 'rb') as file:
    rf_fertility = pickle.load(file)

with open('rf_regular_cycle_model.pkl', 'rb') as file:
    rf_regular_cycle = pickle.load(file)

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
cycle_length = st.number_input(
    "Cycle Length (Days)", 
    min_value=1, 
    value=28, 
    help="Enter the total length of your cycle in days, including the days of menstruation and other phases."
)
average_cycle_length = st.number_input(
    "Average Cycle Length (Days)", 
    min_value=1, 
    value=28, 
    help="Provide the average cycle length based on your past tracked cycles. This helps identify consistency."
)
ovulation_day = st.number_input(
    "Estimated Ovulation Day", 
    min_value=1, 
    max_value=31, 
    value=14, 
    help="Enter the day of your cycle when ovulation typically occurs. This is usually midway through your cycle."
)
luteal_phase_length = st.number_input(
    "Luteal Phase Length (Days)", 
    min_value=1, 
    max_value=18, 
    value=12, 
    help="The number of days between ovulation and the start of your next period."
)
high_fertility_start = st.number_input(
    "High Fertility Start (Days)", 
    min_value=1, 
    max_value=31, 
    value=12, 
    help="Enter the day of your cycle when high fertility typically begins."
)
peak_cycle = st.selectbox(
    "Peak Cycle (Yes=1, No=0)", 
    [1, 0], 
    help="Indicate if this cycle has a noticeable peak fertility phase (1 for Yes, 0 for No)."
)
body_mass_index = st.number_input(
    "Body Mass Index (BMI)", 
    min_value=10.0, 
    max_value=50.0, 
    value=22.0, 
    step=0.1, 
    help="Your body mass index, a measure of body fat based on height and weight."
)
reproductive_status = st.selectbox(
    "Reproductive Status (Fertile=1, Not Fertile=0)", 
    [1, 0], 
    help="Select your current reproductive status: Fertile (1) or Not Fertile (0)."
)

# Ensure the input data matches the exact feature names
feature_columns = [
    "Cycle Length", "Ovulation Day", "Luteal Phase Length", 
    "Average Cycle Length", "Peak Cycle", "Body Mass Index", 
    "Reproductive Status", "High Fertility Start"
]

# Create a DataFrame for predictions
input_data = pd.DataFrame({
    "Cycle Length": [cycle_length],
    "Ovulation Day": [ovulation_day],
    "Luteal Phase Length": [luteal_phase_length],
    "Average Cycle Length": [average_cycle_length],
    "Peak Cycle": [peak_cycle],
    "Body Mass Index": [body_mass_index],
    "Reproductive Status": [reproductive_status],
    "High Fertility Start": [high_fertility_start]
})[feature_columns]  # Ensure the column order matches exactly

# Dynamic Prediction Feedback
st.write("### Predictions:")

# Fertility Prediction with Feedback
if st.button("Get Fertility Prediction"):
    fertility_prediction = rf_fertility.predict(input_data)[0]
    fertility_message = "High Fertility" if fertility_prediction == 1 else "Low Fertility"
    fertility_color = "#2A9D8F" if fertility_prediction == 1 else "#A8DADC"
    
    st.markdown(f"""
        <div style="background-color:{fertility_color}; padding:10px; text-align:center; color:white; font-size:22px; font-weight:bold;">
            Fertility Status: {fertility_message}
        </div>
    """, unsafe_allow_html=True)

# Cycle Regularity Prediction with Feedback
if st.button("Get Cycle Regularity Prediction"):
    cycle_regular_status = rf_regular_cycle.predict(input_data)[0]
    irregularity_message = "Irregular Cycle" if cycle_regular_status == 1 else "Regular Cycle"
    irregularity_color = "#457B9D" if cycle_regular_status == 1 else "#2A9D8F"
    
    st.markdown(f"""
        <div style="background-color:{irregularity_color}; padding:10px; text-align:center; color:white; font-size:22px; font-weight:bold;">
            {irregularity_message}
        </div>
    """, unsafe_allow_html=True)

# Section for asking questions to Gemini with a nice prompt
st.write("### Ask Gemini about your cycle:")
user_question = st.text_input("Ask a question about your cycle", help="Type your question here.")

if user_question:
    # Use Gemini to generate a response to the user’s question
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(user_question)
        st.write(f"**Gemini's Response:** {response.text}")
    except Exception as e:
        st.write(f"Error occurred: {e}")

# Footer
st.markdown("""
    <footer style="text-align:center; padding: 20px; font-size: 14px; background-color:#A8DADC; color:white;">
        Powered by Streamlit & Gemini AI. Created with ❤️ for women's health.
    </footer>
""", unsafe_allow_html=True)





