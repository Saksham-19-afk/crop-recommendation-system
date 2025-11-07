import streamlit as st
import pandas as pd
import joblib 
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import sort


model = joblib.load("crop_app.pkl")

st.title("AI - Based Crop Recommendation System")

st.write("Enter your soil and weather details below:")

N=st.number_input("Nitrogen Levels (kg/ha)",min_value=0,max_value=200,value=50)
P=st.number_input("Phosphorus Levels (kg/ha)",min_value=0,max_value=200,value=50)
K = st.number_input("Potassium Levels (kg/ha)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)


if st.button("Predict Crop"):
    res= [[N,P,K,temperature,humidity,ph,rainfall]]
    prediction = model.predict(res)
    p=model.predict_proba(res)
    st.markdown(f"The crop that you should grow is: <b style='color:green;'> {prediction[0].capitalize()}</b>",unsafe_allow_html=True)
   
