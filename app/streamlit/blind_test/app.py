

import streamlit as st
import pandas as pd

import app.spotify.spotify as spo


st.markdown('# Input tracks')

# Load jazz.csv

input_tracks_jazz = pd.read_csv('./app/streamlit/blind_test/jazz.csv')
st.table(input_tracks_jazz[['track_id', 'artists', 'album_name', 'track_name', 'track_genre']])


# show Spotify player

token = spo.get_access_token()
playlist_id = spo.create_playlist(token)
spo.add_tracks_to_playlist(token, playlist_id, input_tracks_jazz['track_id'])
spo.display_player(playlist_id)


st.markdown('# Models')
