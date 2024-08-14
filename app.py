# Deployment Sales Prediction

import streamlit as st
import numpy as np
import joblib  # or pickle

# Load the model
model = joblib.load('linear_regression_model.pkl')

# Streamlit app
st.title('Sales Prediction App')

# Get user inputs
TV_input = st.text_input('Enter TV Advertising Budget:')
Radio_input = st.text_input('Enter Radio Advertising Budget:')
Newspaper_input = st.text_input('Enter Newspaper Advertising Budget:')

# Convert inputs to numeric, handling potential conversion errors
try:
    TV = float(TV_input)
    Radio = float(Radio_input)
    Newspaper = float(Newspaper_input)
except ValueError:
    st.error("Please enter valid numeric values.")

# Predict button
if st.button('Predict'):
    # Ensure inputs are numeric
    if isinstance(TV, (int, float)) and isinstance(Radio, (int, float)) and isinstance(Newspaper, (int, float)):
        prediction = model.predict(np.array([[TV, Radio, Newspaper]]))
        st.write(f'The predicted sales is: {prediction[0]}')
    else:
        st.error("Make sure all inputs are numeric.")
