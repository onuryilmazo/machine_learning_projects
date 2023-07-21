import pandas as pd 
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        # Veri ön işleme işlemleri burada yapılabilir
        # ...
        pass

    def calculate_distance(self):
        # Mesafe hesaplama işlemleri burada yapılabilir
        R = 6371

        def deg_to_rad(degrees):
            return degrees * (np.pi/180)

        def distcalculate(lat1, lon1, lat2, lon2):
            d_lat = deg_to_rad(lat2 - lat1)
            d_lon = deg_to_rad(lon2 - lon1)
            a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        self.data['distance'] = np.nan
        for i in range(len(self.data)):
            self.data.loc[i, 'distance'] = distcalculate(self.data.loc[i, 'Restaurant_latitude'], 
                                                        self.data.loc[i, 'Restaurant_longitude'], 
                                                        self.data.loc[i, 'Delivery_location_latitude'], 
                                                        self.data.loc[i, 'Delivery_location_longitude'])

    def visualize_data(self):
        pass


class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = None

    def prepare_data(self):
        self.data_processor.load_data()
        self.data_processor.preprocess_data()
        self.data_processor.calculate_distance()

    def split_data(self, test_size=0.10, random_state=42):
        x = np.array(self.data_processor.data[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]])
        y = np.array(self.data_processor.data[["Time_taken(min)"]])
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return xtrain, xtest, ytrain, ytest

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=(3, 1)))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.summary()

    def compile_and_train(self, xtrain, ytrain, batch_size=1, epochs=9):
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs)


class DeliveryTimePredictor:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def predict_delivery_time(self):
        print("Food Delivery Time Prediction")
        a = int(input("Age of Delivery Partner: "))
        b = float(input("Ratings of Previous Deliveries: "))
        c = int(input("Total Distance: "))

        features = np.array([[a, b, c]])
        predicted_time = self.model_trainer.model.predict(features)
        print("Predicted Delivery Time in Minutes =", predicted_time[0][0])


if __name__ == "__main__":
    data_processor = DataProcessor("deliverytime.txt")
    model_trainer = ModelTrainer(data_processor)

    model_trainer.prepare_data()
    xtrain, xtest, ytrain, ytest = model_trainer.split_data()

    model_trainer.create_model()
    model_trainer.compile_and_train(xtrain, ytrain)

    predictor = DeliveryTimePredictor(model_trainer)
    predictor.predict_delivery_time()
