"""The App."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import keras
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('data/AEP_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime')
ts = df[df['Datetime']>='2017'].reset_index(drop=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(ts[['AEP_MW']])

st.write("""
# LSTM Forecasting App for Energy Consumption Data
""")

st.write("""
### Data format and must be greater than 24 hours (timestamps)
| Datetime            | AEP_MW        |
| ------------------- |:-------------:|
| 2017-01-01 00:00:00 | 13240.0       |
| 2017-01-01 01:00:00 | 12876.0       |
| 2017-01-01 02:00:00 | 12591.0	      |
""")

uploaded_file = st.file_uploader("Choose a file", type='csv')

if uploaded_file is not None:  

    df = pd.read_csv(uploaded_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    df['scaled'] = scaler.transform(df[['AEP_MW']])
    
    # Load the model from the file
    model = keras.models.load_model('Forecasting_AEP_MW')
    
    window_size = 24
    sub_seq, next_values = [], []
    for i in range(len(df[['scaled']])-window_size):  
        sub_seq.append(df[['scaled']][i:i+window_size])
        next_values.append(df['scaled'][i+window_size])
    X = np.array(sub_seq)
    y = np.array([next_values]).T
    
    abs_error = np.abs(y - model.predict(X))

    threshold = 0.0890438496934155

    test_score_df = pd.DataFrame()
    test_score_df['Datetime'] = df['Datetime'][window_size:].reset_index(drop=True)
    test_score_df['AEP_MW'] = df['AEP_MW'][window_size:].reset_index(drop=True)
    test_score_df['loss'] = abs_error
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    
    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]

    st.write("Visualize Detected Anomalies from Data")  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_score_df['Datetime'], y=test_score_df['AEP_MW'], name='AEP_MW'))
    fig.add_trace(go.Scatter(x=anomalies['Datetime'], y=anomalies['AEP_MW'], mode='markers', name='Anomaly'))
    fig.update_layout(showlegend=True, title='Detected anomalies')
    st.plotly_chart(fig)
  

