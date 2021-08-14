"""The App."""

import pandas as pd
import numpy as np
import streamlit as st
import keras
import pickle
import plotly.graph_objects as go

# Create sliding window
def sliding_window(seq, window_size):
    sub_seq, next_values = [], []
    for i in range(len(seq)-window_size):
        sub_seq.append(seq[i:i+window_size])
        next_values.append([seq[i+window_size]])
    X = np.array(sub_seq)
    y = np.array(next_values)
    return X,y

window_size = 10

# Load the model from the file
model = keras.models.load_model('anomaly_detection')

# load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

threshold = 18.97461056150496

st.write("""
# LSTM Anomaly Detection App for Web Traffic Data
""")

st.write("""
### Data format and must be greater than 10 timestamps
| timestamp  | value   |
| -----------|:-------:|
| 1          | 10      |
| 2          | 7       |
| 3          | 17      |
""")

uploaded_file = st.file_uploader("Choose a file", type='csv')

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df['scaled'] = scaler.transform(df[['value']])
    
    X, y = sliding_window(df['scaled'], window_size)
    
    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    predict = scaler.inverse_transform(model.predict(X))
    y = scaler.inverse_transform(y)
    
    abs_error = np.abs(y - predict)

    test_score_df = pd.DataFrame()
    test_score_df['timestamp'] = df['timestamp'][window_size:]
    test_score_df['value'] = df['value'][window_size:]
    test_score_df['loss'] = abs_error
    test_score_df['threshold'] = threshold
    test_score_df['anomaly_hat'] = 0
    test_score_df.loc[test_score_df.loss >= test_score_df.threshold, 'anomaly_hat'] = 1
    
    anomalies = test_score_df.loc[test_score_df['anomaly_hat'] == 1]

    st.write("Visualize Detected Anomalies from Data")  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_score_df['timestamp'], y=test_score_df['value'], name='value'))
    fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['value'], mode='markers', name='Anomaly'))
    fig.update_layout(showlegend=True, title='Detected anomalies')
    st.plotly_chart(fig)