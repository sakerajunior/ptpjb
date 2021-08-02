"""The App."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import keras

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
    st.write("SELECT FORECAST PERIOD")
    periods_input = 24*st.number_input('How many days forecast do you want? (integer)')
    
    if periods_input > 0:
        df = pd.read_csv(uploaded_file)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime')
        ts = df[-24*7:].reset_index(drop=True)
        # Load the model from the file
        model = keras.models.load_model('Forecasting_AEP_MW')
        n_future = int(periods_input)
        for i in range(n_future):
            X_future = scaler.transform(ts[['AEP_MW']][-24:]).reshape(1,24,1)
            new_y_hat = scaler.inverse_transform(model.predict(X_future))[0,0]
            new_time = ts['Datetime'].iloc[-1] + pd.DateOffset(hours=1)
            ts.loc[len(ts)] = [new_time,new_y_hat]

        st.write("VISUALIZE FORECASTED DATA")  
        st.write("""
        The following plot shows the last seven days of data and the future predicted value.
        """)
        
        plt.figure(figsize=(15,8))
        sns.lineplot(data=ts[:-n_future],x='Datetime', y='AEP_MW')
        sns.lineplot(data=ts[-n_future-1:],x='Datetime', y='AEP_MW')
        st.pyplot()
  

