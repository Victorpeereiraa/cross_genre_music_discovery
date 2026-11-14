import streamlit as st
from spotipy.oauth2 import SpotifyPKCE, CacheFileHandler


'''
    exec: python app/spotify/bootstrap_spotify.py
    save the refresh token in secrets.toml
'''

auth_manager = SpotifyPKCE(
    client_id=st.secrets["SPOTIFY_CLIENT_ID"],
    redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope="playlist-modify-public playlist-modify-private",
    cache_handler=CacheFileHandler(
        cache_path=".spotify_cache"
    ),
)

# Première connexion interactive
auth_manager.get_access_token()

# Récupérer le refresh token enregistré par Spotipy
token_info = auth_manager.cache_handler.get_cached_token()

print("Refresh token :", token_info["refresh_token"])
