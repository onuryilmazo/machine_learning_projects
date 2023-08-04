import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv") # bozuk veri kümesi 
print(data.head())
print(data.isnull().sum())

# "?" işareti içeren sütunlar için ortalama değeri bulup "?" değerlerini dolduruyoruz
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data[column] = data[column].fillna(data[column].mean())

data = data.dropna()  # Null değerli satırları sildik

print(data.head())

x = np.array(data[["Day", "Month", "ForecastWindProduction", "SystemLoadEA",
                   "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity",
                   "ActualWindProduction", "SystemLoadEP2"]])
y = np.array(data["SMPEP2"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(xtrain, ytrain)

# Predict
features = np.array([[10, 12, 54.10, 4241.05, 49.56, 9.0, 14.8, 491.32, 54.0, 4426.84]])

print(model.predict(features))
