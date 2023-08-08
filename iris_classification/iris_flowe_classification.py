import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

iris = pd.read_csv("data.csv")

print(iris.head())
print(iris.describe())

x = iris.drop("species", axis=1)
y = iris["species"]

xtrain, xtest, ytrain, ytest = train_test_split(x,y , test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(xtrain,ytrain)

#finding optimum k(n_neigbors) value:
'''
k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
x= scaler.fit_transform(x)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, x, y, cv=5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")

plt.show()
#optimum k = 6 
'''


#predict section

x_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(x_new)

print("Prediction: {}".format(prediction))