import requests
from spotipy.oauth2 import SpotifyPKCE, CacheFileHandler
from IPython.display import HTML, display
import pandas as pd



CLIENT_ID = "f1f9e4ee251c4d0993747e73c48d6cce"
PLAYLIST_ID = "33aWOeIiteZ2thsep1QGzN"



"""
How to use Spotify
    1. Get access token
    2. Create a playlist OR reuse the one that is listed above
    3. Add the tracks to the playlist
    4. Show the player with the playlist
"""


def get_access_token():
    """
    Connects to the Cross-Genre Music Discovery app and requests an access token
    """

    auth_manager = SpotifyPKCE(
        client_id="f1f9e4ee251c4d0993747e73c48d6cce",
        redirect_uri="http://127.0.0.1:9090/callback",
        scope="playlist-modify-public",
        cache_handler=CacheFileHandler(
            cache_path=".spotify_token_cache"
        )
    )

    token = auth_manager.get_access_token()

    return token


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
    """
        Shows the Spotify player with the playlist
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
