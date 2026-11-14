from sklearn.mixture import GaussianMixture
from app.preprocessing.preprocessing import preprocessing , PCA_transform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv("../cross_genre_music_discovery/raw_data/spotify-tracks-dataset-detailed.csv")


def atribuir_clusters(
# Recebe um dataframe com as músicas imputadas, o dicionário com o modelo de gmm , \\
    # as features e o nome da coluna que será adicionado no datafram
    df_musicas,
    config,
    features,
    cluster_col="cluster_gmm"
):
    """
    Atribui clusters a músicas cujas features
    já estão escaladas.

    Retorna:
    - DataFrame com a coluna de clusters
    - Coordenadas no espaço usado pelo modelo
    """

    # 1. Recuperar as features já escaladas
    X_scaled = df_musicas[features].to_numpy()

    # 2. Aplicar o PCA treinado, quando existir
    if config.get("pca") is not None:
        X_modelo = config["pca"].transform(X_scaled)
    else:
        X_modelo = X_scaled

    # 3. Identificar o cluster de cada música
    clusters = config["model"].predict(X_modelo)

    # 4. Adicionar os clusters ao DataFrame
    resultado = df_musicas.copy()
    resultado[cluster_col] = clusters

    # 5. Retornar os dados e as coordenadas
    return resultado, X_modelo


def find_medoids(input_df, X_input, cluster_col="cluster_gmm"):

    # Criar um DataFrame com as coordenadas das músicas
    df_coordenadas = pd.DataFrame(
        X_input,
        index=input_df.index
    )

    medoides = []

    # Percorrer os clusters presentes no input
    for cluster, grupo in input_df.groupby(cluster_col):

        # Selecionar as coordenadas das músicas do grupo
        coordenadas_grupo = df_coordenadas.loc[grupo.index]

        # Calcular as distâncias entre as músicas do grupo
        distancias = pairwise_distances(
            coordenadas_grupo,
            metric="euclidean"
        )

        # Somar as distâncias de cada música às demais
        soma_distancias = distancias.sum(axis=1)

        # Encontrar a posição da música mais central
        posicao_medoide = soma_distancias.argmin()

        # Recuperar o registro da música escolhida
        medoide = grupo.iloc[posicao_medoide].copy()

        # Guardar também o tamanho do grupo
        medoide["n_musicas"] = len(grupo)

        medoides.append(medoide)

    return pd.DataFrame(medoides).reset_index(drop=True)


#encontra os gêneros das músicas do input e retorna um set
def get_input_genres(input_df):

    generos_input = set()

    for lista_generos in input_df["genres"]:
        generos_input.update(lista_generos)

    return generos_input

# encontra as músicas presentes nos clusters do input

def get_candidates(catalogo, medoides_df, cluster_col="cluster_gmm"):

    clusters = medoides_df[cluster_col].unique()

    candidatas = catalogo[
        catalogo[cluster_col].isin(clusters)
    ].copy()

    return candidatas


# Função que recebe as músicas dos clusters do input e considera apenas as com gêneros diferentes

def filtrar_generos(candidatas_df, generos_input):

    mascara = candidatas_df["genres"].apply(
        lambda lista: set(lista).isdisjoint(generos_input)
    )

    return candidatas_df[mascara].copy()



def calcular_distancias(
    candidatas_df,
    medoides_df,
    X_candidatas,
    X_input,
    input_df,
    cluster_col="cluster_gmm"
):

    # Garantir que as coordenadas tenham os índices corretos
    coord_candidatas = pd.DataFrame(
        X_candidatas,
        index=candidatas_df.index
    )

    coord_input = pd.DataFrame(
        X_input,
        index=input_df.index
    )

    resultado = candidatas_df.copy()
    resultado["distancia_medoide"] = np.nan

    # Percorrer cada medoide
    for _, medoide in medoides_df.iterrows():

        cluster = medoide[cluster_col]

        # Selecionar as candidatas pertencentes ao mesmo cluster
        grupo_candidatas = resultado[
            resultado[cluster_col] == cluster
        ]

        if grupo_candidatas.empty:
            continue

        # Coordenadas das candidatas do cluster
        X_grupo = coord_candidatas.loc[grupo_candidatas.index]

        # Coordenadas do medoide
        X_medoide = coord_input.loc[
            input_df["track_id"] == medoide["track_id"]
        ]

        # Calcular a distância de cada candidata ao medoide
        distancias = pairwise_distances(
            X_grupo,
            X_medoide,
            metric="euclidean"
        ).ravel()

        # Armazenar as distâncias
        resultado.loc[
            grupo_candidatas.index,
            "distancia_medoide"
        ] = distancias

    return resultado



def recomendar_musicas(
    candidatas_df,
    medoides_df,
    modo="seguro",
    random_state=42
):
    # Gerador de números aleatórios reproduzíveis
    rng = np.random.default_rng(random_state)

    # Quantidade de músicas do input em cada cluster
    contagens = medoides_df.set_index("cluster_gmm")["n_musicas"]

    # Nosso MVP recebe 10 músicas e recomenda 20
    if contagens.sum() != 10:
        raise ValueError("O input precisa ter 10 músicas.")

    if candidatas_df["track_id"].duplicated().any():
        raise ValueError("Existem track_id duplicados.")

    # Cada música do input dá direito a 2 recomendações
    cotas = contagens * 2

    # Quantas candidatas existem em cada cluster?
    capacidades = (
        candidatas_df.groupby("cluster_gmm")
        .size()
        .reindex(cotas.index, fill_value=0)
    )

    # Impedir cotas maiores que a quantidade disponível
    cotas = pd.Series(
        np.minimum(cotas.to_numpy(), capacidades.to_numpy()),
        index=cotas.index
    )

    # Redistribuir vagas que ficaram sem candidatas
    faltantes = 20 - int(cotas.sum())

    while faltantes > 0:
        houve_redistribuicao = False

        for cluster in contagens.sort_values(ascending=False).index:
            if cotas[cluster] < capacidades[cluster]:
                cotas[cluster] += 1
                faltantes -= 1
                houve_redistribuicao = True

            if faltantes == 0:
                break

        if not houve_redistribuicao:
            raise ValueError("Não existem 20 candidatas suficientes.")

    # Selecionar recomendações dentro de cada cluster
    recomendacoes = []

    for cluster, quantidade in cotas.items():

        if quantidade == 0:
            continue

        grupo = candidatas_df[
            candidatas_df["cluster_gmm"] == cluster
        ].copy()

        # Probabilidades definidas pelo modo escolhido
        probabilidades = calcular_probabilidades(
            grupo["distancia_medoide"].to_numpy(),
            modo=modo
        )

        # Sortear músicas sem repetição
        indices = rng.choice(
            len(grupo),
            size=int(quantidade),
            replace=False,
            p=probabilidades
        )

        recomendacoes.append(grupo.iloc[indices])

    # Juntar as recomendações dos clusters
    playlist = pd.concat(recomendacoes, ignore_index=True)

    # Embaralhar a ordem final
    playlist = playlist.iloc[
        rng.permutation(len(playlist))
    ].reset_index(drop=True)

    return playlist
