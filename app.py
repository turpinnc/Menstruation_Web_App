from joblib import load

# Load the trained model
rf_clf_irregular_balanced = load('rf_model_irregular.joblib')

import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from dotenv import load_dotenv
load_dotenv()  # Make sure this is at the beginning of the script to load the environment variables
import warnings
from dotenv import load_dotenv
import os

# Suppress warnings
warnings.filterwarnings("ignore")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])



# Load the trained model
rf_clf_irregular_balanced = load('rf_model_irregular.joblib')

# Setup for the Streamlit app
st.title("Cycle Prediction and Feedback with Gemini")

# User input for Cycle Number, Cycle Length, Ovulation Day
cycle_number = st.number_input("Cycle Number", min_value=1, value=10)
cycle_length = st.number_input("Cycle Length (Days)", min_value=1, value=28)
ovulation_day = st.number_input("Estimated Ovulation Day", min_value=1, max_value=31, value=14)

# Create a DataFrame to hold the user inputs
input_data = pd.DataFrame({
    "Cycle Number": [cycle_number],
    "Cycle Length": [cycle_length],
    "Ovulation Day": [ovulation_day]
})

# **Fertility Prediction**:
def predict_fertility(ovulation_day):
    # Fertility is high if ovulation is between days 12 and 16
    if 12 <= ovulation_day <= 16:
        return "High Fertility"
    else:
        return "Low Fertility"

# Predict Fertility based on Ovulation Day
fertility_prediction = predict_fertility(ovulation_day)

# Visualize Fertility Prediction using a Pie Chart
fig, ax = plt.subplots()
ax.pie([1 if fertility_prediction == "High Fertility" else 0, 1 if fertility_prediction == "Low Fertility" else 0],
       labels=["High Fertility", "Low Fertility"], autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF5733"])
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Display the fertility prediction
st.write(f"Fertility Prediction: {fertility_prediction}")

# Predict button for cycle irregularity
if st.button('Predict Irregular Cycle'):
    # Make prediction using the loaded model
    model_prediction = rf_clf_irregular_balanced.predict(input_data)

    if model_prediction == 1:
        st.write("Prediction: Irregular Cycle")
    else:
        st.write("Prediction: Regular Cycle")

    # Visualize the cycle prediction using a bar chart
    fig, ax = plt.subplots()
    ax.bar(['Regular Cycle', 'Irregular Cycle'], [255, 30], color=['#FF5733', '#4CAF50'])
    ax.set_title("Prediction Counts")
    st.pyplot(fig)

# Section for asking questions to Gemini
user_question = st.text_input("Ask a question about your cycle")

if user_question:
    # Use Gemini to generate a response to the user’s question
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    try:
        response = model.generate_content(user_question)
        st.write("Gemini's Response:", response.text)
    except Exception as e:
        st.write(f"Error occurred: {e}")
