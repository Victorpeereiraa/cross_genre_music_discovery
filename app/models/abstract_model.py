from abc import ABC, abstractmethod



class AbstractModel(ABC):

    audio_features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "valence",
        "tempo"
    ]

    input_size = 10
    output_size = 20

    @property
    @abstractmethod
    def name(self) -> str:
        pass

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
