import pandas as pd 
import numpy as np
import plotly.express as px


data = pd.read_csv("deliverytime.txt")
print(data.head(10))
print(data.info())
print(data.isnull().sum())


# Set the earth's radius (in kilometers)
R = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Function to calculate the distance between two points using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Calculate the distance between each pair of points
data['distance'] = np.nan

for i in range(len(data)):
    data.loc[i, 'distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])
 
figure = px.scatter(data_frame = data, 
                    x="distance",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    trendline="ols", 
                    title = "Relationship Between Distance and Time Taken")
figure.show()

figure = px.scatter(data_frame = data, 
                    x="Delivery_person_Age",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Age")
figure.show()

figure = px.scatter(data_frame = data, 
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Ratings")
figure.show()

fig = px.box(data, 
             x="Type_of_vehicle",
             y="Time_taken(min)", 
             color="Type_of_order")
fig.show()


from sklearn.model_selection import train_test_split

x = np.array(data[["Delivery_person_Age",
                   "Delivery_person_Ratings",
                   "distance"]])
y = np.array(data[["Time_taken(min)"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.10,
                                                random_state=42)  
#test_size: Test verilerinin oranını belirler. Örneğin, 0.10, veri kümesinin %10'unu test için, %90'ını eğitim için kullanacağı anlamına gelir.
#random_state: Rastgele veri bölünmesini kontrol eder. Bu, aynı random_state değeri kullanıldığında, her seferinde aynı veri noktalarının test ve eğitim alt kümelerine atanmasını sağlar.
#Bu sayede, sonuçların tekrarlanabilir olmasını sağlar. Bu değer isteğe bağlıdır, ancak belirli bir değeri kullanmak sonuçların tekrar edilebilir olmasını sağlar.


from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
#LSTM: Uzun Kısa Dönemli Hafıza (Long Short-Term Memory) katmanıdır.
#128: Katmandaki LSTM hücrelerinin sayısı (nöron sayısı).
#return_sequences=True: True olarak ayarlandığında, bu katmanın çıkışı bir sonraki LSTM katmanına beslenmek üzere zaman serisi olarak şekillenir.
#Bu, sırasıyla daha fazla LSTM katmanı ekleyerek modelde çeşitli zaman serisi katmanları oluşturulmasını sağlar.
#input_shape=(xtrain.shape[1], 1): Giriş verilerinin şeklini belirtir. 
# Burada xtrain verilerinin ikinci boyutu (sütun sayısı) 1 olarak belirtilir, 
# çünkü LSTM katmanları zaman serisi verilerini kabul etmek için 3 boyutlu giriş beklerler (örnek sayısı, zaman adımı, özellik sayısı).
model.add(LSTM(64, return_sequences=False))
#64: İkinci LSTM katmanındaki hücrelerin sayısı.
#return_sequences=False: Bu sefer False olarak ayarlandığından, bu katmanın çıkışı bir sonraki katmana sadece son zaman adımındaki çıktıları verir.
model.add(Dense(25)) #tamamen bağlı (dense) katman
#Ardından, bu LSTM katmanlarından oluşan modelimize bir tamamen bağlı (dense) katman ekliyoruz:
#25 katmandaki nöron sayısı
model.add(Dense(1))
#1: Çıkış katmanındaki nöron sayısı. Burada, tek bir değer çıkış verisi olacağı varsayılır.
model.summary()

model.compile(optimizer="adam", loss="mean_squared_error") #derlemesi
#optimizer="adam": Adam optimizasyon algoritması kullanılacağını belirtir. 
# Adam, sık kullanılan ve etkili bir gradyan iniş algoritmasıdır, genellikle hızlı ve verimli eğitim sağlar.
#loss="mean_squared_error": Ortalama karesel hata (mean squared error) kayıp fonksiyonunun kullanılacağını belirtir. 
#Ortalama karesel hata, tahmin edilen değerlerle gerçek değerler arasındaki farkın karesinin ortalamasını hesaplar.
#Bu kayıp fonksiyonu, regresyon problemleri için yaygın olarak kullanılır
model.fit(xtrain, ytrain, batch_size=1, epochs=9) #eğitimi işlemi
#xtrain: Eğitim verileri (giriş özellikleri) numpy dizisi şeklinde verilir.
#ytrain: Eğitim verilerine karşılık gelen hedef çıktı değerleri (etiketler) numpy dizisi şeklinde verilir.
#batch_size=1: Modelin her bir eğitim adımında kullanacağı veri örnekleri sayısıdır. 
#Burada "1" olarak belirtilmiş, yani stokastik gradyan iniş (SGD - stochastic gradient descent) kullanılarak eğitim yapılacağı anlamına gelir. 
#Bir adımın tamamlanması için bir veri örneği kullanılacaktır. Batch size değerini artırarak mini-batch veya tam batch yöntemleri de kullanılabilir.
#epochs=9: Tüm veri kümesi üzerinden modelin kaç kez eğitileceğini belirler. 9 dönemde tüm veri kümesi üzerinde eğitim yapılacak demektir.

print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))