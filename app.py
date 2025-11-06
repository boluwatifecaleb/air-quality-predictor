# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the Trained Model and Feature List ---

# Use Streamlit's caching decorator to load the model only once.
@st.cache_resource
def load_model():
    """Loads the pre-trained Random Forest model."""
    try:
        model = joblib.load('air_quality_predictor.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'air_quality_predictor.pkl' not found. Ensure it's in the same directory.")
        return None

rf_model = load_model()

# List of ALL 33 features the model was trained on (13 current + 20 lagged)
# This list MUST match the columns of the X_train DataFrame used in Colab.
ALL_FEATURE_COLUMNS = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
    'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 
    'CO(GT)_LAG_1H', 'CO(GT)_LAG_6H', 'CO(GT)_LAG_12H', 'CO(GT)_LAG_24H',
    'T_LAG_1H', 'T_LAG_6H', 'T_LAG_12H', 'T_LAG_24H',
    'RH_LAG_1H', 'RH_LAG_6H', 'RH_LAG_12H', 'RH_LAG_24H',
    'NOx(GT)_LAG_1H', 'NOx(GT)_LAG_6H', 'NOx(GT)_LAG_12H', 'NOx(GT)_LAG_24H',
    'C6H6(GT)_LAG_1H', 'C6H6(GT)_LAG_6H', 'C6H6(GT)_LAG_12H', 'C6H6(GT)_LAG_24H'
]

# --- 2. Define Streamlit Interface (UI) ---

st.title('💨 24-Hour CO Exceedance Early Warning System')
st.markdown("Enter the current and recent sensor readings below to predict the probability of a critical CO exceedance (>$4.700$ ppm) 24 hours from now.")

if rf_model:
    # --- 3. User Input Section ---
    st.header('Input Data: Current State and Momentum')

    # Use Streamlit columns to organize inputs for a better look
    col1, col2 = st.columns(2)

    # Dictionary to hold all input values
    user_inputs = {}
    
    # --- Col 1: Current Readings (Top 5 Important) ---
    with col1:
        st.subheader("Current (t) Sensor Readings")
        user_inputs['CO(GT)'] = st.number_input('CO(GT) (Current)', min_value=0.0, max_value=12.0, value=2.0, step=0.1, help="Carbon Monoxide level (ppm)")
        user_inputs['PT08.S5(O3)'] = st.number_input('PT08.S5(O3) (Current)', min_value=500.0, max_value=2500.0, value=1200.0, help="Ozone Sensor Response")
        user_inputs['C6H6(GT)'] = st.number_input('C6H6(GT) (Current)', min_value=0.0, max_value=50.0, value=10.0, help="Benzene Concentration")
        user_inputs['NOx(GT)'] = st.number_input('NOx(GT) (Current)', min_value=0.0, max_value=1500.0, value=200.0, help="Nitrogen Oxides Concentration")
        user_inputs['T'] = st.number_input('Temperature (T)', min_value=0.0, max_value=40.0, value=20.0, step=0.1, help="Ambient Temperature")

    # --- Col 2: Lagged Readings (Top Momentum & Cycle Clues) ---
    with col2:
        st.subheader("Recent (Lagged) Readings")
        user_inputs['CO(GT)_LAG_1H'] = st.number_input('CO(GT) 1 Hour Ago', min_value=0.0, max_value=12.0, value=1.9, step=0.1, help="CO level from 1 hour ago (momentum)")
        user_inputs['NOx(GT)_LAG_24H'] = st.number_input('NOx(GT) 24 Hours Ago', min_value=0.0, max_value=1500.0, value=180.0, help="NOx level from the same time yesterday (daily cycle)")
        user_inputs['PT08.S1(CO)'] = st.number_input('PT08.S1(CO) (Current)', min_value=500.0, max_value=2000.0, value=1300.0, help="CO Sensor Response (Current)")
        user_inputs['RH'] = st.number_input('Relative Humidity (RH)', min_value=10.0, max_value=90.0, value=50.0, step=0.1, help="Current Relative Humidity")
        user_inputs['RH_LAG_1H'] = st.number_input('RH 1 Hour Ago', min_value=10.0, max_value=90.0, value=49.0, step=0.1, help="RH level from 1 hour ago")


    # --- 4. Prediction Logic ---

    # Define a button to trigger the prediction
    if st.button('Predict 24-Hour Exceedance Risk'):
        
        # A. Create a DataFrame filled with default/mean values for ALL 33 features
        # This is CRITICAL: The input features MUST match the training features exactly.
        default_data = {
            'PT08.S2(NMHC)': 1000.0, 'NMHC(GT)': 100.0, 'PT08.S3(NOx)': 1000.0, 
            'NO2(GT)': 100.0, 'PT08.S4(NO2)': 1500.0, 'AH': 0.8,
            # Fill all required lag columns with a reasonable default
            **{col: 0.0 for col in ALL_FEATURE_COLUMNS if 'LAG' in col and col not in user_inputs}
        }
        
        # B. Overwrite default values with the 10 user-provided inputs
        final_input = {**default_data, **user_inputs}
        
        # C. Ensure the DataFrame has the correct columns in the correct order
        input_df = pd.DataFrame([final_input], columns=ALL_FEATURE_COLUMNS)

        # Make the prediction
        # predict_proba returns the probability for [Class 0, Class 1]
        prediction_proba = rf_model.predict_proba(input_df)[0]
        prob_exceed = prediction_proba[1] # Probability of Exceedance (Class 1)

        # --- 5. Display Results ---
        
        st.subheader("Prediction Result")
        
        # Define the confidence levels for the warning
        if prob_exceed >= 0.5:
            st.error(f"⚠️ HIGH RISK: {prob_exceed*100:.1f}% Probability")
            st.metric("24-Hour Forecast", "Critical Exceedance Predicted", delta_color="off")
            st.warning("Immediate action or increased monitoring is recommended. The model shows a high chance of a critical CO level (>4.700 ppm) tomorrow.")
        elif prob_exceed >= 0.25:
            st.warning(f"🟡 MODERATE RISK: {prob_exceed*100:.1f}% Probability")
            st.metric("24-Hour Forecast", "Elevated Risk", delta_color="off")
            st.info("The risk is above baseline. Continue to monitor conditions and lag factors closely.")
        else:
            st.success(f"🟢 LOW RISK: {prob_exceed*100:.1f}% Probability")
            st.metric("24-Hour Forecast", "Normal Conditions Predicted", delta_color="off")
            st.info("Current sensor readings suggest a low probability of a critical event 24 hours from now.")
            
        st.markdown(f"---")
        st.caption(f"Model Confidence for Normal Conditions (0): {prediction_proba[0]*100:.1f}%")