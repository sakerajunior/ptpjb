"""The App."""

import pandas as pd
import numpy as np
import streamlit as st
from tensorflow import keras
import pickle
from sklearn import datasets

# Load the model from the file
model = keras.models.load_model('NN_reg')

# load the scaler
scalerX = pickle.load(open('scalerX.pkl', 'rb'))
scalerY = pickle.load(open('scalerY.pkl', 'rb'))

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")

boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', df.CRIM.min(), df.CRIM.max(), df.CRIM.mean())
    ZN = st.sidebar.slider('ZN', df.ZN.min(), df.ZN.max(), df.ZN.mean())
    INDUS = st.sidebar.slider('INDUS', df.INDUS.min(), df.INDUS.max(), df.INDUS.mean())
    CHAS = st.sidebar.slider('CHAS', df.CHAS.min(), df.CHAS.max(), df.CHAS.mean())
    NOX = st.sidebar.slider('NOX', df.NOX.min(), df.NOX.max(), df.NOX.mean())
    RM = st.sidebar.slider('RM', df.RM.min(), df.RM.max(), df.RM.mean())
    AGE = st.sidebar.slider('AGE', df.AGE.min(), df.AGE.max(), df.AGE.mean())
    DIS = st.sidebar.slider('DIS', df.DIS.min(), df.DIS.max(), df.DIS.mean())
    RAD = st.sidebar.slider('RAD', df.RAD.min(), df.RAD.max(), df.RAD.mean())
    TAX = st.sidebar.slider('TAX', df.TAX.min(), df.TAX.max(), df.TAX.mean())
    PTRATIO = st.sidebar.slider('PTRATIO', df.PTRATIO.min(), df.PTRATIO.max(), df.PTRATIO.mean())
    B = st.sidebar.slider('B', df.B.min(), df.B.max(), df.B.mean())
    LSTAT = st.sidebar.slider('LSTAT', df.LSTAT.min(), df.LSTAT.max(), df.LSTAT.mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

X = user_input_features()
X_scaled = scalerX.transform(X)

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(X)
st.write('---')

# Apply Model to Make Prediction
prediction = scalerY.inverse_transform(model.predict(X_scaled))

st.header('Prediction of PRICE')
st.write(prediction)
st.write('---')