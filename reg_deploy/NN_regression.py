"""The App."""

import pandas as pd
import numpy as np
import streamlit as st
from tensorflow import keras
import pickle
import base64

# Load the model from the file
model = keras.models.load_model('NN_reg')

# load the scaler
scalerX = pickle.load(open('scalerX.pkl', 'rb'))
scalerY = pickle.load(open('scalerY.pkl', 'rb'))

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")

uploaded_file = st.file_uploader("Choose a file", type='csv')

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df)    
        
        # Apply Model to Make Prediction
        X_scaled = scalerX.transform(df)
        prediction = scalerY.inverse_transform(model.predict(X_scaled))
        df['price_prediction']=prediction

        st.header('Prediction of PRICE')
        st.write(df[['price_prediction']])

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  
        link= f'<a href="data:file/csv;base64,{b64}" download="price_prediction.csv">Download</a>'
        st.markdown(link, unsafe_allow_html=True)
    except:
        st.write("# Error!!!")