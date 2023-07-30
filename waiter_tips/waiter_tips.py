import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("tips.csv")
print(data.head())

# Verilerde amacımız tipi bulmak  bunla diğer veriler arasında bir korelasyona bakılabilir ama bunu için sayısal veriler lazım 
# Bunun için şimdi string ifadeleri sayısal verilere döndüreceğiz

data["sex"] = data["sex"].map({"Female":0, "Male":1})
data["smoker"] = data["smoker"].map({"No":0, "Yes":1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch":0, "Dinner":1})

print(data.head())

print(data.corr()["tip"])

# We will split the data

x = np.array(data[["total_bill", "sex", "smoker", "day", "time", "size"]])
y = np.array(data["tip"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)

print(model.score(xtest, ytest)) #0.4429399687489901 // çok düşük

# prediction

feature = np.array([[18.08, 1, 0, 3, 1, 2]])
xy = model.predict(feature)
features = np.array([[18.08, 0, 0, 3, 1, 2]])
xx = model.predict(features)

print(xx , xy)
