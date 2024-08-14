# Deployment Sales Prediction
import sklearn
import streamlit as st
import numpy as np
import pickle  

# Load the model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
# Streamlit app
st.title('Sales Prediction App')

# Get user inputs
TV = st.number_input('TV Advertising Budget', min_value=0.0)
Radio = st.number_input('Radio Advertising Budget', min_value=0.0)
Newspaper = st.number_input('Newspaper Advertising Budget', min_value=0.0)


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
