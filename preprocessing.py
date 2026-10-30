import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("/home/victor/code/Victorpeereiraa/cross_genre_music_discovery/spotify-tracks-dataset-detailed.csv")

def preprocessing(df):

    df = df.copy()

    features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "valence",
        "tempo"
    ]

    metadata = [
        "track_id",
        "artists",
        "track_name"
    ]

    # 1. Remover comedy
    df = df[df["track_genre"] != "comedy"].copy()

    # 2. Selecionar colunas necessárias
    df = df[metadata + ["track_genre"] + features]

    # 3. Remover valores nulos
    df = df.dropna(subset=metadata + features)

    # 4. Remover duplicatas exatas
    df = df.drop_duplicates()

    # 5. Preservar todos os gêneros de cada música
    genres = (
        df.groupby("track_id")["track_genre"]
        .agg(lambda x: sorted(set(x)))
    )

    # 6. Manter uma linha por música
    df = df.drop_duplicates(subset=["track_id"]).copy()

    # 7. Adicionar gêneros associados
    df["genres"] = df["track_id"].map(genres)

    # 8. Separar metadados e features
    df_metadata = df[metadata + ["genres"]].reset_index(drop=True)
    X = df[features].reset_index(drop=True)

    # 9. Escalonar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df_metadata, scaler
