import numpy as np 
import pandas as pd
from autots import AutoTS

data = pd.read_csv("AAPL.csv")

print(data.head())

model = AutoTS(forecast_length=5, frequency="infer", ensemble="simple")
model = model.fit(data, date_col="Date", value_col= "Close", id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)

