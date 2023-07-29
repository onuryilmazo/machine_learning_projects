import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Transform categorical columns to numerical
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
data["isFraud"] = data["isFraud"].map({"No Fraud": 0, "Fraud": 1})

x = data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
y = data["isFraud"]

# Split the data into smaller sample size for faster performance
x_sample, _, y_sample, _ = train_test_split(x, y, train_size=0.1, random_state=42)

# Use RandomForestClassifier model with reduced complexity
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(x_sample, y_sample)

#accuracy = model.score(x_sample, y_sample)
#print("Model Accuracy:", accuracy)
#Model Accuracy: 0.9991622947779374


# Perform k-fold cross-validation (k=5) in parallel using all available CPU cores
k = 5
scores = cross_val_score(model, x, y, cv=k, n_jobs=-1)

# Calculate mean accuracy
mean_accuracy = np.mean(scores)

print("Mean Accuracy with {}-fold cross-validation: {:.4f}".format(k, mean_accuracy))

#Cross-validation, makine öğrenimi modellerinin performansını değerlendirmek için kullanılan istatistiksel bir yöntemdir. Modelin genelleştirme yeteneğini değerlendirmek ve overfitting (aşırı uyum) gibi problemleri tespit etmek için kullanılır.
#Cross-validation işlemi, veri kümesini daha küçük alt kümeler (katmanlar) halinde böler ve bu alt kümeleri tekrar tekrar kullanarak eğitim ve test setleri oluşturur. Her iterasyonda, farklı alt kümeler eğitim ve test için kullanılır ve modelin performansı bu iterasyonların ortalaması alınarak değerlendirilir.
#En yaygın kullanılan cross-validation yöntemi "k-fold cross-validation" olarak bilinir. Bu yöntemde veri kümesi k adet alt küme (katman) halinde bölünür. Ardından, her bir katmanı sırayla test seti olarak kullanarak modeli eğitir ve performansını ölçeriz. Sonuçlar kere kere alındıktan sonra ortalaması alınarak modelin genel performansı değerlendirilir.
#K-fold cross-validation avantajları şunlardır:
#Daha güvenilir performans ölçümü: Modelin gerçek veri kümesindeki performansını daha iyi tahmin etmek için daha fazla veri kullanılır.
#Overfitting'i tespit etme: Daha küçük veri kümesi üzerinde aşırı uyuma eğilimi olan modeller, cross-validation ile fark edilerek düzeltilir.
#Veri kullanımı: Veri kümesinin tamamı hem eğitim hem de test için kullanılır, böylece verinin daha etkin kullanımı sağlanır.
#K-fold cross-validation kullanarak modelin performansını daha güvenilir bir şekilde değerlendirebilir ve model seçimi, hiperparametre ayarlaması gibi işlemlerde daha iyi kararlar alabilirsiniz.

# Make predictions
features = np.array([[4, 9000.60, 9000.60, 0.0]])
prediction = model.predict(features)
if prediction[0] == 0:
    print("Prediction: No Fraud")
else:
    print("Prediction: Fraud")

#Prediction: No Fraud

