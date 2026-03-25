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
    res = [[N, P, K, temperature, humidity, ph, rainfall]]

    prediction = model.predict(res)[0]
    probabilities = model.predict_proba(res)[0]

    # Get the highest probability 
    max_prob = max(probabilities)
    confidence = round(max_prob * 100, 2)

    # Suitability levels
    if confidence < 25:
        suitability = " Very Low Suitability"
        color = "red"
        message = " The model strongly suggests NOT growing this crop in current conditions."
    elif confidence < 50:
        suitability = " Low Suitability"
        color = "orange"
        message = " The crop can grow, but the conditions are not favorable."
    elif confidence < 75:
        suitability = " Moderate Suitability"
        color = "yellow"
        message = " Conditions are acceptable, but not ideal."
    else:
        suitability = " High Suitability"
        color = "green"
        message = " Excellent conditions! You can confidently grow this crop."

    # Main output
    st.markdown(
        f"""
        <div style="padding:15px;border-radius:10px;background-color:#111;">
            <h2 style="color:{color};">{suitability}</h2>
            <p style="color:white;font-size:18px;">{message}</p>
            <h3 style="color:lightgreen;">Recommended Crop: <b>{prediction.capitalize()}</b></h3>
            <p style="color:white;">Confidence: <b style="color:{color};">{confidence}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability bar chart
    st.subheader(" Crop Suitability Breakdown")

    prob_df = pd.DataFrame({
        'Crop': model.classes_,
        'Probability (%)': probabilities * 100
    }).sort_values(by="Probability (%)", ascending=False)

    st.bar_chart(prob_df.set_index("Crop"))
