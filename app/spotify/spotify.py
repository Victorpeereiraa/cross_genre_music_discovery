import requests
from spotipy.oauth2 import SpotifyPKCE, CacheFileHandler
import pandas as pd
import streamlit as st


CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
REDIRECT_URI = st.secrets.get("SPOTIPY_REDIRECT_URI")
REFRESH_TOKEN = st.secrets.get("SPOTIFY_REFRESH_TOKEN")



"""
How to use Spotify
    1. Get access token
    2. Create a playlist
    3. Add the tracks to the playlist
    4. Show the player with the playlist
"""


def get_access_token():
    """
    Connects to the Cross-Genre Music Discovery app of Spotify and requests an access token
    """

    auth_manager = SpotifyPKCE(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope="playlist-modify-public",
        cache_handler=CacheFileHandler(
            cache_path=".spotify_token_cache"
        ),
        open_browser=False
    )

    # use the refresh token to get an access token
    token_info = auth_manager.refresh_access_token(REFRESH_TOKEN)

    return token_info["access_token"]


def create_playlist(access_token):
    """
    Creates a playlist in the Cross-Genre Music Discovery
    """

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://api.spotify.com/v1/me/playlists",
        headers=headers,
        json={
            "name": "Cross-Genre Music Discovery",
            "description": "My music recommendations",
            "public": True
        },
        timeout=20
    )

    response.raise_for_status()

    playlist_id = response.json()["id"]

    return playlist_id


def empty_playlist(access_token, playlist_id):
    """
        Removes all tracks from the playlist
    """

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.put(
        f"https://api.spotify.com/v1/playlists/{playlist_id}/items",
        headers=headers,
        json={"uris": []}
    )

    response.raise_for_status()


def add_tracks_to_playlist(access_token, playlist_id, track_ids):
    """
        Add the tracks to the playlist
    """

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Build tracks URIs
    track_uris = [
        f"spotify:track:{track_id}"
        for track_id in track_ids
    ]

    # Add to playlist
    response = requests.post(
        f"https://api.spotify.com/v1/playlists/{playlist_id}/items",
        headers=headers,
        json={"uris": track_uris},
        timeout=20
    )

    response.raise_for_status()


def display_player(playlist_id):
    from IPython.display import HTML, display
    """
        Shows the Spotify player with the playlist
        Use in Notebook
    """

    display(HTML(f"""
        <iframe
        src="https://open.spotify.com/embed/playlist/{playlist_id}"
        width="100%"
        height="400"
        frameborder="0"
        allow="autoplay; clipboard-write; encrypted-media;
                fullscreen; picture-in-picture"
        loading="lazy">
        </iframe>
        """))


def display_player_streamlit(playlist_id):
    """
    Shows the Spotify player with the playlist in Streamlit.
    Use in Streamlit page
    """
    url = f"https://open.spotify.com/embed/playlist/{playlist_id}"

    st.iframe(
        src=url,
        height=600,
        scrolling=False
    )
