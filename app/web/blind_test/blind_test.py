
import logging
import sys
import random
from uuid import uuid4

import streamlit as st
import pandas as pd
from pathlib import Path

import app.spotify.spotify as spo
from app.web.blind_test.scoring import save_vote
from app.models.baseline_model import BaselineModel
from app.models.autoencoder_knn_model import Autoencoder_kNN_Model
from app.models.gmm_recommender import GMMModel
from app.models.dbscan_model import DBSCANModel
from app.models.kmeans_model import KMeansModel



st.set_page_config(
    page_title="Beyond Genre",
    layout="wide"
)

# ------------------------ Input track lists ------------------------

input_tracklists = [
    {
        "name": "Eclética",
        "filename": "Playlist_ECLETICA.csv"
    },
    {
        "name": "Homogênea",
        "filename": "Playlist_HOMOGENEA.csv"
    },
    {
        "name": "Intermediária",
        "filename": "Playlist_INTERMEDIARIA.csv"
    }
]

# shuffle and store the input tracklists in the session
if "input_tracklists" not in st.session_state:
    random.shuffle(input_tracklists)
    st.session_state.input_tracklists = input_tracklists

# use the input_tracklists stored in the session
input_tracklists = st.session_state.input_tracklists

# ------------------------ Logs ------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------ Creating (empty) Spotify playlists ------------------------

token = spo.get_access_token()


if "input_playlist_ids" not in st.session_state:
    st.session_state.input_playlist_ids = [
        spo.create_playlist(token)
        for _ in range(len(input_tracklists))
    ]

input_playlist_ids = st.session_state.input_playlist_ids


nb_models = 6
if "recommendation_playlist_ids" not in st.session_state:
    st.session_state.recommendation_playlist_ids = [
        spo.create_playlist(token)
        for _ in range(nb_models)
    ]

recommendation_playlist_ids = st.session_state.recommendation_playlist_ids


# ------------------------ Load input tracks ------------------------

root = Path(__file__).resolve().parents[3]
path = root / "artifacts" / "input_track_lists"

# Load CSV files
for tracklist in input_tracklists:
    if "track_df" not in tracklist:
        tracklist["track_df"] = pd.read_csv(
            path / tracklist["filename"]
        )

# Initialize Spotify playlists
if not st.session_state.get("input_playlists_initialized", False):
    for i, tracklist in enumerate(input_tracklists):
        spo.empty_playlist(token, input_playlist_ids[i])
        spo.add_tracks_to_playlist(
            token,
            input_playlist_ids[i],
            tracklist["track_df"]["track_id"]
        )

    st.session_state.input_playlists_initialized = True

# always make sure we use the input_tracklists stored in the session
input_tracklists = st.session_state.input_tracklists

# show Spotify input playlists
st.markdown('# Input tracks')

columns = st.columns(len(input_tracklists))
for i in range(len(columns)):
    with columns[i]:

        label = chr(65 + i)  # A, B, C...
        input_tracklists[i]['label'] = label
        st.subheader(f"Playlist {label}")

        # show Spotify player
        spo.display_player_streamlit(input_playlist_ids[i])


# Show radio buttons to select the input tracklist
track_tracklist_names = tuple(
    tracklist["label"] for tracklist in input_tracklists
)

# clean trial if user changes input tracklist
def on_input_change():
    st.session_state.trial = None

selected_input_list_label = st.radio(
    'Select a playlist',
    track_tracklist_names,
    key="selected_input_playlist",
    on_change=on_input_change
)


# ------------------------ Load models ------------------------
# Add here as many models as you want

if "models" not in st.session_state:
    models = []

    # Load baseline
    baseline_model = BaselineModel()
    models.append(baseline_model)

    # # Load autoencoder
    autoencoder_knn_model = Autoencoder_kNN_Model()
    models.append(autoencoder_knn_model)

    # Load gmm models
    pca3_gmm_model = GMMModel(configuracao="pca3_tied15")
    models.append(pca3_gmm_model)

    pca5_gmm_model = GMMModel(configuracao="pca5_diag30")
    models.append(pca5_gmm_model)

    dbscan_model = DBSCANModel()
    models.append(dbscan_model)

    kmeans_model = KMeansModel()
    models.append(kmeans_model)

    # Shuffling the list so that we don't know which model produce which recommendation
    random.shuffle(models)

    # store the models in the session
    st.session_state.models = models

# be sure we always use the models stored in the session
models = st.session_state.models



# ------------------------ Recommend ------------------------
# Call each model and ask for a recommendation

st.markdown('# Recommend')
recommendations = []

if "trial" not in st.session_state:
    st.session_state.trial = None

disabled=st.session_state.trial is not None

if st.button('Recommend', disabled = disabled):

    results = []
    input_tracks = next(
        (tracklist for tracklist in input_tracklists \
            if tracklist["label"] == selected_input_list_label),
        None
    )

    # calling all models to get recommendations
    for model in models:

        recommendations = model.recommend(
            input_track_ids=input_tracks['track_df']['track_id']
        )

        # limit to 10 tracks
        recommendations = recommendations[:10]

        results.append({
            "model_id": model.name,
            "track_ids": recommendations
        })

    random.shuffle(results)

    # add recommendations to Spotify playlists
    for i, result in enumerate(results):
        spo.empty_playlist(token, recommendation_playlist_ids[i])
        spo.add_tracks_to_playlist(
            token,
            recommendation_playlist_ids[i],
            result["track_ids"])

    # save test results in the session
    st.session_state.trial = {
        "id": str(uuid4()),
        "input_track_ids": input_tracks['track_df']['track_id'].tolist(),
        "input_track_name": input_tracks['name'],
        "results": results
    }

    # Rerun to refresh button state
    st.rerun()


# ------------------------ Show results and Score models ------------------------

# retrieve the test results
trial = st.session_state.trial

if trial is not None:

    n = len(trial["results"])
    columns = st.columns(n)

    # Show recommendations
    for i, result in enumerate(trial["results"]):
        with columns[i]:

            label = chr(65 + i)  # A, B, C...
            st.subheader(f"Playlist {label}")

            # show Spotify player
            spo.display_player_streamlit(recommendation_playlist_ids[i])


    # Scoring form
    with st.form("scoring_form"):

        name = st.text_input("Enter your name")
        ranks = []

        # scoring
        for i in range(n):

            label = chr(65 + i) # A, B, C...

            rank = st.selectbox(
                f"Score Playlist {label}",
                options=[None] + list(range(1, n + 1)),
                format_func=lambda x: (
                    "Choose position" if x is None else str(x)
                ),
                key=f"{trial['id']}_{i}"
            )

            ranks.append(rank)

        # saving score
        submitted = st.form_submit_button("Save scores")

        if submitted:

            # check form is not empty
            if not name.strip():
                st.error("Please enter your name")

            elif None in ranks or len(set(ranks)) != n:
                st.error("Please rank each playlist")

            else:

                # build the score json
                scored_results = []

                for i, result in enumerate(trial["results"]):
                    scored_results.append({
                        **result,
                        "blind_label": chr(65 + i),
                        "rank": ranks[i]
                    })

                # build the vote json
                vote = {
                    "id": trial["id"],
                    "voter": name.strip(),
                    "input_track_name": trial["input_track_name"],
                    "input_track_ids": trial["input_track_ids"],
                    "results": scored_results
                }

                # save vote json
                try:
                    save_vote(vote)

                except Exception as e:
                    st.error(f"Unable to save. Please try again. {e}")
                    st.exception(e)

                else:
                    st.session_state.trial = None
                    st.success("Vote saved !")
