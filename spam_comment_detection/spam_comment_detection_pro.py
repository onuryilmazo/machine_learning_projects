import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

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

def preprocess_data(data):
    if data is None:
        return None

    # Selecting necessary columns
    data = data[["CONTENT", "CLASS"]]

    # Mapping class labels to human-readable form
    data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam"})

    return data

def train_model(x, y):
    cv = CountVectorizer()
    x = cv.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = BernoulliNB()
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    return model, cv, accuracy  # Return the CountVectorizer instance as well

def predict_sample(model, cv, sample):
    data = cv.transform([sample]).toarray()
    prediction = model.predict(data)
    return prediction[0]

if __name__ == "__main__":
    filename = "Youtube01-Psy.csv"

    data = load_data(filename)
    if data is not None:
        print(data.sample(5))

        data = preprocess_data(data)
        if data is not None:
            print(data.sample(5))

            x = np.array(data["CONTENT"])
            y = np.array(data["CLASS"])

            trained_model, cv, accuracy = train_model(x, y)  # Store the CountVectorizer instance in 'cv'
            print("Model Accuracy:", accuracy)

            sample1 = "Suna bi bak: https://pornhub.com/"
            sample2 = "Bir daha video cekme sen bok gibi olmus"

            prediction1 = predict_sample(trained_model, cv, sample1)
            prediction2 = predict_sample(trained_model, cv, sample2)

            print("Sample 1 Prediction:", prediction1)
            print("Sample 2 Prediction:", prediction2)
