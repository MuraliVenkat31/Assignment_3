import pandas as pd
import numpy as np
import string
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder	

st.title('Car Price Prediction ML Model')
st.image("Car Image.jpg")
st.set_option('deprecation.showfileUploaderEncoding',False) 

with open('xgb_model.pkl', "rb") as fr:
	xgb_model = pickle.load(fr)

df = pd.read_csv("df.csv")
	
with st.sidebar:
	st.sidebar.image("image.jpg", use_column_width = True)
	city = st.selectbox('Select city', df['city'].unique())
	fuel_type = st.selectbox('fuel_type', df['fuel_type'].unique())
	kms_driven = st.slider('No of Kms Driven', 11,20000)
	transmission = st.selectbox('Transmission Type', df['transmission'].unique())
	ownerNo = st.selectbox('Number of Owners', df['ownerNo'].unique())
	model = st.selectbox('Select Car Brand', df['model'].unique())
	model_year = st.slider('Car Manufactured Year', 1994,2024)
	mileage = st.slider('Car Mileage', 10,40)
	Engine = st.slider('Engine CC', 700,5000)
	insurance_validity = st.selectbox('select type of Insurance', df['insurance_validity'].unique())
	body_type = st.selectbox('body type', df['body_type'].unique())

if st.button("Predict Price"):
	input_data = pd.DataFrame([[city,fuel_type,kms_driven,transmission,ownerNo,model,model_year,mileage,Engine,insurance_validity,body_type]],
		columns=['city','fuel_type','kms_driven','transmission','ownerNo','model','model_year','mileage','Engine','insurance_validity','body_type'])
	# encoding "model" Column
	input_data.replace({'model':{'Maruti':0,'Ford':1,'Tata':2,'Hyundai':3,'Jeep':4,'Datsun':5,'Honda':6,'Mahindra':7,
 				'Renault':8,'Mercedes-Benz':9,'Audi':10,'BMW':11,'Toyota':12,'Mini':13,'Kia':14,'Skoda':15,'Volkswagen':16, 
				'Nissan':17,'Fiat':18,'Mahindra Ssangyong':19,'Mitsubishi':20,'Jaguar':21,'Land Rover':22,'Chevrolet':23,
				'Mahindra Renault':24,'Volvo':25,'Isuzu':26,'Lexus':27,'Porsche':28}},inplace=True)

	# encoding "Fuel_Type" Column
	input_data.replace({'fuel_type':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}},inplace=True)

	# encoding "Transmission" Column
	input_data.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)

	# encoding "city" Column
	input_data.replace({"city":{'Bangalore':0, 'chennai':1, 'delhi':2, 'Hyderabad':3, 'Jaipur':4, 'Kolkata':5}},inplace=True)

	#encoding "insurance_validity" Column
	input_data.replace({"insurance_validity":{'Third Party insurance':0, 'Comprehensive':1, 'Third Party':2, 'Zero Dep':3}},inplace=True)

	# Encoding "body_type" Column
	input_data.replace({"body_type":{'Hatchback':0,'SUV':1, 'Sedan':2, 'MUV':3, 'Convertibles':4, 'Minivans':5, 'Coupe':6, 'Wagon':7,
				 'Pickup Trucks':8}},inplace=True)

	st.table(input_data)
	predicted_price = xgb_model.predict(input_data)
	st.write(f"Predicted Price:${predicted_price[0]:.2f}")

