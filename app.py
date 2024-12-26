import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('simple_linear_model.pkl')

# App title
st.title("House Price Prediction App")
st.write("This app predicts house prices based on the square footage of the property.")

# User input for square footage
area = st.number_input("Enter the square footage of the house:", min_value=300, max_value=10000, step=10)

# Predict button
if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame({'area': [area]})
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.write(f"Estimated House Price: **${prediction:,.2f}**")
