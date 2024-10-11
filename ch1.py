#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

model = load_model('california_housing_model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

st.title("California Housing Price Prediction")

st.write("Enter the values for the following features to predict the median house value.")

inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter {feature}:", min_value=0.0)

input_df = pd.DataFrame([inputs.values()], columns=feature_names)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.write(f"Predicted Median House Value: ${prediction[0][0] * 100000:.2f}")

