import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
print(data.head())

x = np.array(data["Text"])  # Sadece ["Text"] kolonunu alın
y = np.array(data["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

user = input("Enter a text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
