import numpy as np
import pandas as pd 
import plotly.graph_objects as go
data = pd.read_csv("TTMI.csv")
print(data.head())

figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], high=data["High"],
                                        low=data["Low"], close=data["Close"])])
figure.update_layout(title = "Tata Motors Stock Price Analysis", xaxis_rangeslider_visible=False)
figure.show()


# Date sütununu çıkartıyoruz
#data = data.drop(columns=["Date"])
#print(data.corr())

#Autots, ARIMA, ETS, Prophet, TBATS, ve LSTM gibi farklı zaman serisi modellerini destekler ve 
# bu modellerin hiperparametrelerini otomatik olarak ayarlayarak en iyi sonuçları elde etmeye çalışır.

from autots import AutoTS
model = AutoTS(forecast_length=5, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)


'''
After 10 minutes this is the output 
                Close
2023-07-31  14.074892
2023-08-01  14.050239
2023-08-02  14.097482
2023-08-03  14.118558
2023-08-04  13.999468
'''