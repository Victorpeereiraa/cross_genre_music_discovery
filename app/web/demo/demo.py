
import logging
import sys

# Streamlit
import streamlit as st

# Spotify player
import app.spotify.spotify as spo

# Model
from app.models.autoencoder_knn_model import Autoencoder_kNN_Model

# Track selection
from app.models.track_selection import TrackSelection



# ------------------------ Configs ------------------------

st.set_page_config(
    page_title="Beyond Genre",
    layout="wide"
)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------ Creating (empty) Spotify playlists ------------------------
# Store Spotify playlists in Session to avoid recreating them at every page refresh

token = spo.get_access_token()


if "input_playlist" not in st.session_state:
    st.session_state.input_playlist = spo.create_playlist(token)

input_playlist = st.session_state.input_playlist


if "recommendation_playlist" not in st.session_state:
    st.session_state.recommendation_playlist = spo.create_playlist(token)

recommendation_playlist = st.session_state.recommendation_playlist



# ------------------------ Load Model ------------------------
# Store model in Session to avoid recreating it at every page refresh

if "model" not in st.session_state:
    model = Autoencoder_kNN_Model()
    st.session_state.model = model

model = st.session_state.model



# ------------------------ Init Session ------------------------

# we will later store the input tracks in the session
if "input_track_ids" not in st.session_state:
    st.session_state.input_track_ids = []

# Track if the input tracks have changed since last recommendation
if "input_tracks_have_changed" not in st.session_state:
    st.session_state.input_tracks_have_changed = True

# store recommendation in Session
if "recommended_track_ids" not in st.session_state:
    st.session_state.recommended_track_ids = []



# ------------------------ Select input tracks ------------------------

# retrieve the input tracks from Session
input_track_ids = st.session_state.input_track_ids


# Show input tracks selection form
selection = TrackSelection()


if input_track_ids:
    for track_id in input_track_ids:
        selection.add(track_id)


# Render selection
input_track_ids = selection.render()


# update Session
if input_track_ids:
    st.session_state.input_track_ids = input_track_ids
    st.session_state.input_tracks_have_changed = True



# ------------------------ Load Input Tracks into Spotify Player ------------------------

# Retrieve info from Session
input_track_ids = st.session_state.input_track_ids
input_tracks_have_changed = st.session_state.input_tracks_have_changed

# If input tracks have changed, update the Spotify playlist
if input_tracks_have_changed and input_track_ids:

    # empty Spotify player
    spo.empty_playlist(
        token,
        input_playlist)

    # add new input tracks to Spotify player
    spo.add_tracks_to_playlist(
        token,
        input_playlist,
        input_track_ids)

# if the input playlist is not empty, display it
if input_track_ids:

    # Show Spotify player
    spo.display_player_streamlit(input_playlist)



# ------------------------ Call Model ------------------------

# Retrieve info from Session
input_track_ids = st.session_state.input_track_ids
input_tracks_have_changed = st.session_state.input_tracks_have_changed

# If input tracks have changed, we can recommend
if input_tracks_have_changed and input_track_ids:

    # If input tracked have not changed, don't allow a new recommendation
    disabled = input_tracks_have_changed is False

    # Show button to call a new recommendation
    if st.button('Recommend', disabled = disabled):

        # Call model
        recommendation = model.recommend(
            input_track_ids=input_track_ids
        )

        # add recommendations to Spotify player
        spo.empty_playlist(
            token,
            recommendation_playlist)

        spo.add_tracks_to_playlist(
            token,
            recommendation_playlist,
            recommendation)

        # Update Session
        st.session_state.recommended_track_ids = recommendation
        st.session_state.input_tracks_have_changed = input_tracks_have_changed


# ------------------------ Show Spotify Player ------------------------

if st.session_state.recommended_track_ids:
    # show Spotify player
    spo.display_player_streamlit(recommendation_playlist)
