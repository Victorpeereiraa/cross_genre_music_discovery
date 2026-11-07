
import random

import streamlit as st
import pandas as pd
from pathlib import Path

import app.spotify.spotify as spo
from app.models.baseline_model import BaselineModel
from app.models.autoencoder_knn_model import Autoencoder_kNN_Model


token = spo.get_access_token()

root = Path(__file__).resolve().parents[3]
path = root / "artifacts" / "input_track_lists"


# ------------------------ Load input tracks ------------------------
# Temporary code until we have final test track lists

st.markdown('# Input tracks')

# Load jazz.csv

input_tracks_jazz = pd.read_csv(path / 'jazz.csv')
#st.table(input_tracks_jazz[['track_id', 'artists', 'album_name', 'track_name', 'track_genre']])


# show Spotify player

playlist_id = spo.create_playlist(token)
spo.add_tracks_to_playlist(token, playlist_id, input_tracks_jazz['track_id'])
spo.display_player_streamlit(playlist_id)


# ------------------------ Load models ------------------------
# Add here as many models as you want

st.markdown('# Models')
models = []

# Load baseline
baseline_model = BaselineModel()
models.append(baseline_model)

# Load autoencoder
autoencoder_knn_model = Autoencoder_kNN_Model()
models.append(autoencoder_knn_model)

st.write(f'Loaded {len(models)} models')
st.write('Model list has been shuffled')

# Shuffling the list so that we don't know which model produce which recommendation
# We probably will have to change this when we will implement the user scoring
random.shuffle(models)


# ------------------------ Recommend ------------------------
# Call each model and ask for a recommendation
# Todo: display each reco in a Spotify player

st.markdown('# Recommend')
recommendations = []

if st.button('Recommend'):

    # calling all models to get recommendations
    for model in models:
        recommendations.append(model.recommend(input_track_ids=input_tracks_jazz['track_id']))

    # print recommendations
    columns = st.columns(len(recommendations))

    for index, recommendation in enumerate(recommendations):

        with columns[index]:

            st.subheader(f"Playlist {index + 1}")

            # show Spotify player
            playlist_id = spo.create_playlist(token)
            spo.add_tracks_to_playlist(token, playlist_id, recommendation)
            spo.display_player_streamlit(playlist_id)



# ------------------------ Score ------------------------
