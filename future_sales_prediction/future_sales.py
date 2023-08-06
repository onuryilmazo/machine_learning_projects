import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{filename}' is empty.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data: {e}")
        return None

def trained_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    return model, accuracy

def predict_sample(model, sample):
    data = np.array(sample, dtype=float)
    prediction = model.predict(data)
    return prediction

def main():
    filename = "https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv"

    data = load_data(filename)
    if data is not None:
        print(data.sample(5))

        x = np.array(data[["TV", "Radio", "Newspaper"]])
        y = np.array(data["Sales"])

        model, accuracy = trained_model(x, y)
        print("Model Accuracy:", accuracy)

        sample1 = np.array([[230.1, 37.8, 69.2]])
        sample2 = np.array([[46.8, 35.0, 65.6]])

        prediction1 = predict_sample(model, sample1)
        prediction2 = predict_sample(model, sample2)

        print("Sample 1 Prediction:", prediction1)
        print("Sample 2 Prediction:", prediction2)

if __name__ == "__main__":
    main()
