import streamlit as st
import requests
import json
import pandas as pd

st.write("""
# LaLiga Match Result Predictor
""")

st.sidebar.header('Input Parameters')

def user_input_features():
    # Collect input from the user
    jornada = st.sidebar.number_input("Jornada", min_value=1, max_value=100, value=1)

    input_dict = {
        "jornada": jornada
    }

    return input_dict

input_dict = user_input_features()

if st.button('Predict'):
    # In frontend/main.py
    response = requests.post(
    #url= "http://localhost:8001/predict",
    url="http://chase-complains-model-container:8000/predict",  # Change 'laliga-model-container' to 'model' http://model:8001/predict
    data=json.dumps(input_dict)
)

    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.write(pd.DataFrame(prediction))
    else:
        st.write("Error en la predicción")