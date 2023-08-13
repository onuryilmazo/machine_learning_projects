import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv("Youtube01-Psy.csv")
print(data.sample(5))

#We only need the content and class column from the dataset for the rest of the task. So let’s select both the columns and move further:

data = data[["CONTENT", "CLASS"]]
print(data.sample(5))

data["CLASS"] = data["CLASS"].map({0: "Not Spam",
                               1: "Spam"})
print(data.sample(5))


#Now let’s move further by training a classification Machine Learning model to classify spam and not spam comments. 
#As this problem is a problem of binary classification, I will use the Bernoulli Naive Bayes algorithm to train the model:

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])
#CountVectorizer, metin verilerini sayısal vektörlere dönüştürmek için kullanılan bir özellik çıkarma (feature extraction) yöntemidir. 
#Özellikle doğal dil işleme (NLP) alanında metin tabanlı veri kümesini makine öğrenimi algoritmalarında kullanılabilir hale getirmek için yaygın olarak kullanılır.

cv = CountVectorizer() 
x = cv.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)
print(model.score(xtest,ytest))

#Now let’s test the model by giving spam and not spam comments as input:

sample = "Suna bi bak: https://pornhub.com/" 
data = cv.transform([sample]).toarray()
print(model.predict(data))

sample = "Bir daha video cekme sen bok gibi olmus" 
data = cv.transform([sample]).toarray()
print(model.predict(data)) 