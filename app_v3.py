import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from dotenv import load_dotenv

# Load environment variables (if needed, but not necessary for Streamlit secrets)
load_dotenv()

# Load the trained model
rf_clf_irregular_balanced = load('rf_model_irregular.joblib')

# Configure the Gemini API using Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Set up page configuration and title
st.set_page_config(
    page_title="Cycle Prediction and Feedback",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the app with custom styling
st.markdown("""
    <h1 style="text-align:center; color:#4CAF50;">Cycle Prediction and Feedback with Gemini</h1>
    <p style="text-align:center; font-size:18px; color:#555;">Track and predict your menstrual cycle with advanced AI.</p>
""", unsafe_allow_html=True)

# Add a banner-like feature without needing an external image
st.markdown(
    """
    <div style="background-color:#4CAF50; padding:10px; text-align:center; color:white; font-size:18px;">
        🩺 Welcome to the Cycle Prediction Tool! 🩺
    </div>
    """, unsafe_allow_html=True
)

# User input for Cycle Number, Cycle Length, Ovulation Day
st.write("### Please enter your cycle details:")
cycle_number = st.number_input("Cycle Number", min_value=1, value=10, help="Enter your cycle number.")
cycle_length = st.number_input("Cycle Length (Days)", min_value=1, value=28, help="Enter the length of your cycle in days.")
ovulation_day = st.number_input("Estimated Ovulation Day", min_value=1, max_value=31, value=14, help="Select the estimated ovulation day.")

# Additional User Inputs
st.write("### Additional Information:")
age = st.number_input("Age", min_value=18, value=30, help="Enter your age.")
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0, step=0.1, help="Enter your weight.")
symptoms = st.multiselect("Symptoms (Select all that apply)", 
                          ["Cramps", "Mood Changes", "Heavy Bleeding", "Irregular Periods", "Other"],
                          help="Select symptoms you may be experiencing during your cycle.")

# Create a DataFrame to hold the user inputs
input_data = pd.DataFrame({
    "Cycle Number": [cycle_number],
    "Cycle Length": [cycle_length],
    "Ovulation Day": [ovulation_day],
    "Age": [age],
    "Weight": [weight],
    "Symptoms": [", ".join(symptoms)]  # Join symptoms into a string for easier processing
})

# Fertility prediction function
def predict_fertility(ovulation_day):
    # Fertility is high if ovulation is between days 12 and 16
    if 12 <= ovulation_day <= 16:
        return "High Fertility"
    else:
        return "Low Fertility"

# Predict fertility based on ovulation day
fertility_prediction = predict_fertility(ovulation_day)

# Visualize fertility prediction using a pie chart with custom colors and labels
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([1 if fertility_prediction == "High Fertility" else 0, 1 if fertility_prediction == "Low Fertility" else 0],
       labels=["High Fertility", "Low Fertility"], autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF5733"])
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Display the fertility prediction with custom styling
st.markdown(f"""
    <h3 style="text-align:center; color:#4CAF50;">Fertility Prediction</h3>
    <p style="text-align:center; font-size:22px; font-weight: bold; color:#4CAF50;">{fertility_prediction}</p>
""", unsafe_allow_html=True)

# Predict button for cycle irregularity with a custom icon and color
if st.button('🔮 Predict Cycle Regularity', key="predict_button"):
    # Make prediction using the loaded model
    model_prediction = rf_clf_irregular_balanced.predict(input_data)

    if model_prediction == 1:
        st.write("**Prediction:** Irregular Cycle")
    else:
        st.write("**Prediction:** Regular Cycle")

    # Visualize the cycle prediction using a bar chart with custom colors
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Regular Cycle', 'Irregular Cycle'], [255, 30], color=['#4CAF50', '#FF5733'])
    ax.set_title("Prediction Counts", fontsize=14, color="#333")
    ax.set_ylabel("Count", fontsize=12, color="#555")
    st.pyplot(fig)

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

# Footer with custom text and branding
st.markdown(
    """
    <footer style="text-align:center; padding: 20px; font-size: 14px; background-color:#4CAF50; color:white;">
        Powered by Streamlit & Gemini AI. Created with ❤️ for women's health.
    </footer>
    """, unsafe_allow_html=True
)
