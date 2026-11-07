
from pathlib import Path
import joblib
import pandas as pd
from keras.models import load_model

from app.models.abstract_model import AbstractModel


class Autoencoder_kNN_Model(AbstractModel):

    def __init__(self):

        # Loading artifacts
        root = Path(__file__).resolve().parents[2]
        path = root / "artifacts" / "models" / "autoencoder_knn"

        self.model = joblib.load(path / "autoencoder_knn_model.joblib")
        self.encoder = load_model(path / "autoencoder_encoder.keras", compile=False)
        self.scaler = joblib.load(path / "autoencoder_knn_scaler.joblib")
        self.catalog_clean = pd.read_parquet(root / "artifacts" / "catalog_clean.parquet").set_index("track_id")


    def recommend(self, input_track_ids):

        # retrieve input tracks from scaled catalog
        input_tracks = self.catalog_clean.loc[input_track_ids, self.audio_features]

        # scale input tracks
        input_tracks_scaled = self.scaler.transform(input_tracks)

        # encode input tracks
        input_tracks_encoded = self.encoder.predict(input_tracks_scaled)

        # calculate input tracks profile
        input_profile = input_tracks_encoded.mean(axis=0).reshape(1, -1)

        # Find neighbors
        distances, indices = self.model.kneighbors(input_profile, n_neighbors=self.output_size)

        output_track_ids = self.catalog_clean.index[indices[0]].tolist()

        return output_track_ids
