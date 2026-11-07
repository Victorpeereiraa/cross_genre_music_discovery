from abc import ABC, abstractmethod



class AbstractModel(ABC):

    audio_features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    input_size = 10
    output_size = 20

    @abstractmethod
    def recommend(
        self,
        track_ids: list[str]
    ) -> list[str]:
        """
        Input:
            User's Spotify track IDs

        Output:
            Spotify track IDs recommended by the model
        """
        pass
