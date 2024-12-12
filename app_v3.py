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
    <h1 style="text-align:center; color:#F4A6D3;">Cycle Prediction and Feedback with Gemini</h1>
    <p style="text-align:center; font-size:18px; color:#555;">Track and predict your menstrual cycle with advanced AI.</p>
""", unsafe_allow_html=True)

# Add a banner-like feature without needing an external image
st.markdown(
    """
    <div style="background-color:#F4A6D3; padding:10px; text-align:center; color:white; font-size:18px;">
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

# **Check the columns of input_data before prediction** (print the columns for debugging)
st.write("Fertility Input Data Columns: ", input_data_fertility.columns)
st.write("Irregularity Input Data Columns: ", input_data_irregularity.columns)

# **Predict fertility** based on the input data
fertility_status = rf_clf_fertility_noisy.predict(input_data_fertility)

# Display the fertility prediction
if fertility_status == 1:
    fertility_message = "High Fertility"
    fertility_color = "#FF66B2"  # Pink color for high fertility
else:
    fertility_message = "Low Fertility"
    fertility_color = "#FF5733"  # Red color for low fertility

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

    if model_prediction == 1:
        st.write("**Prediction:** Irregular Cycle")
    else:
        st.write("**Prediction:** Regular Cycle")

    # Enhanced Bar Chart for Cycle Prediction with better UI
    labels = ['Regular Cycle', 'Irregular Cycle']
    counts = [255, 30]  # Example values, replace with your actual prediction counts

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))  # Increase size for better visibility
    bars = ax.bar(labels, counts, color=['#4CAF50', '#FF5733'], edgecolor='black', linewidth=1.5)

    # Add titles and labels
    ax.set_title("Cycle Prediction Results", fontsize=18, color="#333", fontweight='bold')
    ax.set_xlabel("Cycle Type", fontsize=14, color="#555")
    ax.set_ylabel("Count", fontsize=14, color="#555")

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5,  # Position the label above the bar
                round(yval, 0), ha='center', fontsize=12, color='black', fontweight='bold')

    # Add gridlines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Customize the ticks for a cleaner look
    ax.tick_params(axis='x', labelsize=12, rotation=0)  # X-ticks should be readable without rotation
    ax.tick_params(axis='y', labelsize=12)

    # Set background color to white for clarity
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f8f8')

    # Show the plot in the Streamlit app
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
    <footer style="text-align:center; padding: 20px; font-size: 14px; background-color:#F4A6D3; color:white;">
        Powered by Streamlit & Gemini AI. Created with ❤️ for women's health.
    </footer>
    """, unsafe_allow_html=True
)






