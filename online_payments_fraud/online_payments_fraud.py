import pandas as pd
import numpy as np 
import plotly.express as px 
data = pd.read_csv("PS_20174392719_1491204439457_log.csv")
print(data.head())

type = data["type"].value_counts()
transactions = type.index
quantity = type.values
'''
figure = px.pie(data, 
             values=quantity, 
             names=transactions,
             hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()
'''
#now let's have a look at the correlation between the features of the data with the isFraud column:

#correlation_hatali = data.corr()  #pandas kütüphanesinin methodu corr. Datamızda verilen kolonlar arasındaki korelasyonu buluyor.
#print(correlation["isFraud"].sort_values(ascending=False)) 
#fakat bir hata var corr methodu sayısal değer olan kolanlar arasındaki korelasyonu buluyor sadece. 
#bu nedenle bu şekilde bırakırsak hata alırız

numeric_data = data.select_dtypes(include=[np.number]) #sadece sayısal veriler dahil edildi
correlation = numeric_data.corr()
print(correlation["isFraud"].sort_values(ascending=False))  

'''
isFraud           1.000000
amount            0.076688
isFlaggedFraud    0.044109
step              0.031578
oldbalanceOrg     0.010154
newbalanceDest    0.000535
oldbalanceDest   -0.005885
newbalanceOrig   -0.008148

boyle bir çıktı aldık bunun anlamı şu isFradu kolonu ile eksili olanlar arasında negatif artılı olanlar arasında pozitif bir ilişki var fakat bu ilişki çok zayıf
çünkü iki kolon arasında kuvvetli bir korealson ilişkisi kurulması için değerin 0.6-1 aralığı ya da -06, -1 aralığında olması lazım
'''

# now we will transform categorical columns to numerical 

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

#.map() yöntemi, pandas DataFrame veya Series içindeki değerleri belirli bir sözlükle eşleştirerek değiştirmek için kullanılır. 
# Bu, veri dönüşümü veya kategorik verilerin kodlaması gibi işlemleri kolaylaştırır.

from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

#np.array() fonksiyonu, NumPy kütüphanesindeki bir işlevdir ve verileri NumPy dizisi (array) haline getirmek için kullanılır. 
# Verileri NumPy dizisi olarak dönüştürmenin bazı avantajları vardır:
#NumPy dizileri, liste veya DataFrame'lerden daha verimlidir ve hızlı matematiksel işlemler yapmanıza olanak tanır.
#  Özellikle, makine öğrenimi modellerinin eğitimi gibi büyük veri kümeleri üzerinde çalışırken performansı artırabilir.
#Makine öğrenimi kütüphaneleri (örneğin scikit-learn) genellikle NumPy dizilerini girdi olarak beklerler. 
# Bu nedenle, verileri NumPy dizilerine dönüştürerek, bu kütüphaneleri daha kolay ve doğrudan kullanabilirsiniz.
#NumPy dizileri, çok boyutlu yapılar ve diziler üzerindeki matematiksel işlemler için gelişmiş işlevsellik sağlar.
#Bu, veri hazırlama ve manipülasyon süreçlerini kolaylaştırabilir.

from sklearn.tree import DecisionTreeClassifier

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

from sklearn.model_selection import cross_val_score
k = 2
scores = cross_val_score(model, x, y, cv=k, n_jobs=-1)

# Calculate mean accuracy
mean_accuracy = np.mean(scores)

print("Mean Accuracy with {}-fold cross-validation: {:.4f}".format(k, mean_accuracy))

#print("Model Score: ", model.score(xtest, ytest))
#Model Score:  0.9997375295082843

#prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
#features = np.array([4,9000.60, 9000.60, 0.0])   #bu hata verdi neden? alttakiyle arasındaki farka bak. 
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
#['Fraud']