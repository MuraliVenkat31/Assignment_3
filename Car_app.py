import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load model and data
model = pk.load(open('model.pkl', 'rb'))
df = pd.read_csv("Final_data3.csv")

# App setup
st.title('Car Price Prediction ML Model')

# Input section
st.header('Enter Car Details:')
city = st.selectbox('Select City', df['city'].unique())
model = st.selectbox('Select Car Brand', df['model'].unique())
model_year = st.slider('Car Manufactured Year', 1994, 2024, label='Year')
kms_driven = st.slider('No of Kms Driven', 11, 20000000, label='Kms Driven')
fuel_type = st.selectbox('Fuel Type', df['fuel type'].unique())
transmission = st.selectbox('Transmission Type', df['transmission'].unique())
insurance_validity = st.selectbox('Select Car Insurance', df['insurance_validity'].unique())
mileage = st.slider('Car Mileage', 10, 40, label='Mileage')
ownerNo = st.slider('No of Owner', 10, 40, label='ownerNo')
Engine = st.slider('Engine CC', 700, 5000, label='Engine CC')
body_type = st.selectbox('Select Body Type', df['body_type'].unique())
Age = st.slider('Car Mileage', 10, 40, label='Age')

# Prediction section
if st.button("Predict Price"):
    try:
        input_data = pd.DataFrame({
            'model': [model],
            'modelYear': [model_year],
            'Kms_Driven': [kms_driven],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission],
            'Milage': [mileage],
            'Engine': [Engine],
            'body_type': [body_type],
            'Age': [Age],
            'insurance_validity': [insurance_validity],
            'city': [city],
            'ownerNo': [ownerNo] 
        })
        predicted_price = model.predict(input_data)
        st.write(f"Predicted Price: ${predicted_price[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")\