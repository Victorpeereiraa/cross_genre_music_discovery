
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA


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

    # 4. Remover duplicatas exatas, desconsiderando track_id
    df = df.drop_duplicates(
        subset=df.columns.drop("track_id")
    ).copy()

    # 5. Preservar todos os gêneros de cada música
    genres = (
        df.groupby("track_id")["track_genre"]
        .agg(lambda x: sorted(set(x)))
    )

    # 6. Manter uma linha por música
    df = df.drop_duplicates(subset=["track_id"]).copy()

    # 7. Adicionar gêneros associados
    df["genres"] = df["track_id"].map(genres)

    # 8. Consolidar músicas com mesmo artista, título e features
    df["audio_key"] = list(
        df[features].itertuples(index=False, name=None)
    )

    chaves_generos = (
        df.groupby(["artists", "track_name"])
        .filter(
            lambda x: (
                x["genres"].apply(tuple).nunique() > 1 and
                x["audio_key"].nunique() == 1
            )
        )[["artists", "track_name"]]
        .drop_duplicates()
    )

    df["qtd_generos"] = df["genres"].apply(len)

    df_casos = df.merge(
        chaves_generos,
        on=["artists", "track_name"],
        how="inner",
        sort=False
    )

    df_casos = (
        df_casos
        .sort_values("qtd_generos", ascending=False, kind="stable")
        .drop_duplicates(subset=["artists", "track_name"])
    )

    df = df.merge(
        chaves_generos.assign(consolidar=True),
        on=["artists", "track_name"],
        how="left"
    )

    df = df[df["consolidar"].isna()].drop(columns=["consolidar"])

    df = pd.concat([df, df_casos], ignore_index=True)

    # 9. Remover gêneros fora do escopo
    generos_excluir = {"kids", "children", "sleep"}

    df = df[
        ~df["genres"].apply(
            lambda x: bool(generos_excluir.intersection(x))
        )
    ].copy()

    # 10. Remover colunas auxiliares
    df = df.drop(columns=["audio_key", "qtd_generos"])

    # 11. Separar metadados e features
    df_metadata = df[metadata + ["genres"]].reset_index(drop=True)
    X = df[features].reset_index(drop=True)

    # 12. Escalonar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df_metadata, scaler, features

def PCA_transform(df_scaled, number_of_components = 3):
    # 1. Treinar o modelo com o número de features escolhida para o pca

    pca = PCA(n_components=number_of_components)

    X_proj = pd.DataFrame(
        pca.fit_transform(df_scaled),
        columns=[f"PC{i+1}" for i in range(number_of_components)],
        index=df_scaled.index
    )

    return X_proj, pca
