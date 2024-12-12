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

# Load the fertility and cycle irregularity models
rf_clf_fertility_noisy = load('rf_model_fertility.joblib')  # Model trained with 'Ovulation Day (Noisy)'
rf_clf_irregular_balanced = load('rf_model_irregular.joblib')  # Model trained with 'Ovulation Day'

# Configure the Gemini API using the environment variable
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Set up page configuration and title
st.set_page_config(
    page_title="Cycle Prediction and Feedback",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the app with custom styling
st.markdown("""
    <h1 style="text-align:center; color:#4C8C91;">Cycle Prediction and Feedback with Gemini</h1>
    <p style="text-align:center; font-size:18px; color:#555;">Track and predict your menstrual cycle with advanced AI.</p>
""", unsafe_allow_html=True)

# Privacy and Security Message
st.markdown("""
    <div style="background-color:#FFEB3B; padding:15px; text-align:center; color:#333; font-size:16px;">
        <strong>Privacy and Security Notice:</strong><br>
        We value your privacy. All data entered is processed securely and is not shared with any third parties. 
        Your personal information will never be sold or spread. Your data is used solely to provide personalized cycle predictions and support.
    </div>
""", unsafe_allow_html=True)

# Add a banner-like feature without needing an external image
st.markdown(
    """
    <div style="background-color:#A8DADC; padding:10px; text-align:center; color:white; font-size:18px;">
        🩺 Welcome to the Cycle Prediction Tool! 🩺
    </div>
    """, unsafe_allow_html=True
)

# User input for Cycle Number, Cycle Length, Ovulation Day
st.write("### Please enter your cycle details:")
cycle_number = st.number_input("Cycle Number", min_value=1, value=10, help="Enter your cycle number.")
cycle_length = st.number_input("Cycle Length (Days)", min_value=1, value=28, help="Enter the length of your cycle in days.")
ovulation_day = st.number_input("Estimated Ovulation Day", min_value=1, max_value=31, value=14, help="Select the estimated ovulation day.")

# Create the DataFrame with only the relevant columns for prediction
# Ensure that "Ovulation Day (Noisy)" is used for fertility and "Ovulation Day" for cycle irregularity
input_data_fertility = pd.DataFrame({
    "Cycle Number": [cycle_number],  # Features for fertility model (use "Ovulation Day (Noisy)")
    "Cycle Length": [cycle_length],
    "Ovulation Day (Noisy)": [ovulation_day]  # For fertility model
})

input_data_irregularity = pd.DataFrame({
    "Cycle Number": [cycle_number],  # Features for cycle irregularity model (use "Ovulation Day")
    "Cycle Length": [cycle_length],
    "Ovulation Day": [ovulation_day]  # For irregularity model
})

# **Predict fertility** based on the input data
fertility_status = rf_clf_fertility_noisy.predict(input_data_fertility)

# Display the fertility prediction
if fertility_status == 1:
    fertility_message = "High Fertility"
    fertility_color = "#2A9D8F"  # Calm greenish-blue for high fertility
else:
    fertility_message = "Low Fertility"
    fertility_color = "#A8DADC"  # Light blue for low fertility

# Display the fertility prediction with a colored box
st.markdown(f"""
    <div style="background-color:{fertility_color}; padding:10px; text-align:center; color:white; font-size:22px; font-weight:bold;">
        Fertility Status: {fertility_message}
    </div>
""", unsafe_allow_html=True)

# Alternative representation: A simple colored circle or graphic for fertility
fig, ax = plt.subplots(figsize=(3, 3))
circle = plt.Circle((0.5, 0.5), 0.4, color=fertility_color, ec="black", lw=3)
ax.add_artist(circle)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
st.pyplot(fig)

# Predict button for cycle irregularity with a custom icon and color
if st.button('🔮 Predict Cycle Regularity', key="predict_button"):
    # For cycle irregularity, use the cycle irregularity model
    model_prediction = rf_clf_irregular_balanced.predict(input_data_irregularity)

    # Display the cycle irregularity result in a box with peaceful colors
    if model_prediction == 1:
        irregularity_message = "Irregular Cycle"
        irregularity_color = "#457B9D"  # Peaceful blue for irregular cycle
    else:
        irregularity_message = "Regular Cycle"
        irregularity_color = "#2A9D8F"  # Calm green for regular cycle

    # Display the cycle irregularity prediction with a colored box
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

# Footer with custom text and branding
st.markdown(
    """
    <footer style="text-align:center; padding: 20px; font-size: 14px; background-color:#A8DADC; color:white;">
        Powered by Streamlit & Gemini AI. Created with ❤️ for women's health.
    </footer>
    """, unsafe_allow_html=True
)







