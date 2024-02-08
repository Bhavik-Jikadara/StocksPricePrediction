import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import yfinance as yf


im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon=im
)

st.title('Stock Trend Prediction')

# input the Ticker
start_date = st.date_input('Enter the start Date: ',
                           datetime.datetime(2010, 5, 17))
end_date = st.date_input('Enter the End Date: ')
user_input = st.text_input('Enter the Stock Ticker', 'TTM')
df = yf.download(user_input, start=start_date, end=end_date)


def stock_trend_prediction():
    #  Describing Data
    st.subheader('Data From {} - {}'.format(start_date.year, end_date.year))
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()

    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    # Spliting Data into Training and testing
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))


    # Testing Part
    input_data = scaler.fit_transform(data_testing)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    # Load my model
    model = load_model('model/model.h5')
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scaler_factor = 1 / scaler[0]
    y_predicted = y_predicted * scaler_factor
    y_test = y_test * scaler_factor

    # final Graph
    st.subheader('Predictions VS Original')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)


if st.button('Prediction'):
    stock_trend_prediction()
