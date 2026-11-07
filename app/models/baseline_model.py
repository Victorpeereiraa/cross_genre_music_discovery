
from pathlib import Path
import joblib
import pandas as pd

from app.models.abstract_model import AbstractModel


class BaselineModel(AbstractModel):

    def __init__(self):

        # Loading artifacts
        root = Path(__file__).resolve().parents[2]
        path = root / "artifacts" / "models" / "baseline"

        self.model = joblib.load(path / "baseline_model.joblib")
        self.scaler = joblib.load(path / "baseline_scaler.joblib")
        self.catalog_clean = pd.read_parquet(root / "artifacts" / "catalog_clean.parquet").set_index("track_id")


    def recommend(self, input_track_ids):

        # retrieve input tracks from scaled catalog
        input_tracks = self.catalog_clean.loc[input_track_ids, self.audio_features]

        # scale input tracks
        input_tracks_scaled = self.scaler.transform(input_tracks)

        # calculate input tracks profile
        input_profile = input_tracks_scaled.mean(axis=0).to_frame().T

        # Find neighbors
        distances, indices = self.model.kneighbors(input_profile, n_neighbors=self.output_size)

        output_track_ids = self.catalog_clean.index[indices[0]].tolist()

        return output_track_ids
