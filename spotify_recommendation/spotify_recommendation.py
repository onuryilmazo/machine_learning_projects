import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

class SpotifyRecommender:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.normalized_data = self._normalize_data()
        self.kmeans_model = self._build_kmeans_model()

    def _normalize_data(self):
        datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        normalization_data = self.dataset.select_dtypes(include=datatypes)
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(normalization_data), columns=normalization_data.columns)
        return normalized_data

    def _build_kmeans_model(self):
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(self.normalized_data)
        return kmeans

    def recommend_songs(self, song_name, num_recommendations=5):
        song_name = song_name.lower()
        song = self.dataset[self.dataset.name.str.lower() == song_name].iloc[0]
        song_features = self.normalized_data[self.dataset.name.str.lower() != song_name]

        distances = []
        for song_features in tqdm(song_features.values):
            distance = np.abs(song_features - song.drop(["id", "name", "artists", "release_date", "year"]).values).sum()
            distances.append(distance)

        rec_data = self.dataset[self.dataset.name.str.lower() != song_name].copy()
        rec_data['distance'] = distances
        rec_data = rec_data.sort_values('distance')
        recommended_songs = rec_data[["artists", "name"]].head(num_recommendations)

        return recommended_songs


if __name__ == "__main__":
    dataset_path = "spotify.csv"
    recommender = SpotifyRecommender(dataset_path)

    song_name = input("Enter the song name: ")
    num_recommendations = int(input("Enter the number of recommendations you want: "))

    recommendations = recommender.recommend_songs(song_name, num_recommendations)
    print("\nRecommended Songs:")
    for index, song in recommendations.iterrows():
        print(f"Artist: {song['artists']}, Song Name: {song['name']}")
