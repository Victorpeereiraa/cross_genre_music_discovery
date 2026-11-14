
from pathlib import Path
import joblib
from sklearn.metrics import pairwise_distances
import pandas as pd
from recommender import (
    find_medoids,
    get_input_genres,
    get_candidates,
    filtrar_generos,
    calcular_distancias,
    recomendar_musicas
)


from app.models.abstract_model import AbstractModel


class GMMModel(AbstractModel):

    # Features utilizadas no treinamento do nosso GMM
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

    def __init__(self, modo="seguro", configuracao="pca3_tied15"):

        if modo not in ["seguro", "surpresa"]:
            raise ValueError("Modo inválido.")

        self.modo = modo


        # Encontrar a raiz do projeto
        root = Path(__file__).resolve().parents[2]

        # Localizar o modelo salvo
        configuracoes = {
        "pca3_tied15": "gmm_pca3_tied15.joblib",
        "pca5_diag30": "gmm_pca5_diag30.joblib"
        }

        if configuracao not in configuracoes:
            raise ValueError("Configuração de GMM inválida.")

        caminho_modelo = (
            root / "artifacts" / "models" / "gmm"
            / configuracoes[configuracao]
        )

        # Carregar os objetos treinados
        objetos = joblib.load(caminho_modelo)

        self.model = objetos["model"]
        self.pca = objetos["pca"]
        self.scaler = objetos["scaler"]

        # Carregar o catálogo de músicas
        caminho_catalogo = root / "artifacts" / "catalog_gmm.parquet"

        self.catalogo = pd.read_parquet(caminho_catalogo).set_index("track_id")


        # Selecionar as features
        X = self.catalogo[self.audio_features]

        # Aplicar scaler e preservar os nomes das features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.audio_features,
            index=self.catalogo.index
        )

        # Aplicar PCA e preservar os nomes dos componentes
        X_pca = pd.DataFrame(
            self.pca.transform(X_scaled),
            columns=[
                f"PC{i+1}"
                for i in range(self.pca.n_components_)
            ],
            index=self.catalogo.index
        )

        # Atribuir os clusters
        self.catalogo["cluster_gmm"] = self.model.predict(X_pca)

        # Guardar as coordenadas
        self.coordenadas = X_pca


    def get_input_medoids(self, track_ids):

        # Recuperar as músicas escolhidas
        input_df = self.catalogo.loc[track_ids].reset_index()

        # Recuperar suas coordenadas PCA
        X_input = self.coordenadas.loc[track_ids].to_numpy()

        # Calcular os medoides
        medoides_df = find_medoids(input_df, X_input)

        return medoides_df

    def recommend(self, track_ids: list[str]) -> list[str]:

        # 1. Recuperar as músicas do usuário
        input_df = self.catalogo.loc[track_ids].reset_index()

        # 2. Calcular os medoides
        medoides_df = self.get_input_medoids(track_ids)

        # 3. Identificar os gêneros proibidos
        generos_input = get_input_genres(input_df)

        # 4. Recuperar candidatas dos clusters
        candidatas = get_candidates(
            self.catalogo.reset_index(),
            medoides_df
        )

        # 5. Filtrar os gêneros proibidos
        candidatas = filtrar_generos(
            candidatas,
            generos_input
        )

        # 6. Excluir músicas do input e duplicatas
        candidatas = candidatas[
            ~candidatas["track_id"].isin(track_ids)
        ].drop_duplicates(subset="track_id").copy()

        # 7. Recuperar coordenadas PCA
        X_input = self.coordenadas.loc[track_ids].to_numpy()

        X_candidatas = self.coordenadas.loc[
            candidatas["track_id"]
        ].to_numpy()

        # 8. Calcular distâncias aos medoides
        candidatas = calcular_distancias(
            candidatas_df=candidatas,
            medoides_df=medoides_df,
            X_candidatas=X_candidatas,
            X_input=X_input,
            input_df=input_df
        )

        # 9. Gerar as 20 recomendações
        playlist = recomendar_musicas(
            candidatas,
            medoides_df,
            modo=self.modo
        )

        # 10. Retornar somente os IDs
        return playlist["track_id"].tolist()
