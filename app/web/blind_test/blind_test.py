
import random
from uuid import uuid4

import streamlit as st
import pandas as pd
from pathlib import Path

import app.spotify.spotify as spo
from app.web.blind_test.scoring import save_vote
from app.models.baseline_model import BaselineModel
from app.models.autoencoder_knn_model import Autoencoder_kNN_Model




# ------------------------ Creating (empty) playlists ------------------------

token = spo.get_access_token()

input_playlist_id = spo.create_playlist(token)

nb_models = 2
recommendation_playlist_ids = []
for i in range(nb_models):
    recommendation_playlist_ids.append( spo.create_playlist(token))





root = Path(__file__).resolve().parents[3]
path = root / "artifacts" / "input_track_lists"



# ------------------------ Load input tracks ------------------------
# Temporary code until we have final test track lists

st.markdown('# Input tracks')

# Load jazz.csv
input_tracks_jazz = pd.read_csv(path / 'jazz.csv')


# show Spotify player
spo.add_tracks_to_playlist(token, input_playlist_id, input_tracks_jazz['track_id'])
spo.display_player_streamlit(input_playlist_id)

# st.iframe(
#     "https://open.spotify.com/embed/playlist/1rX1RzugQm6ZAUhTRn4fHy",
#     height=352,
#     width="stretch"
# )



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

if "trial" not in st.session_state:
    st.session_state.trial = None

disabled=st.session_state.trial is not None

if st.button('Recommend', disabled):

    results = []

    # calling all models to get recommendations
    for model in models:
        recommendations = model.recommend(
            input_track_ids=input_tracks_jazz['track_id']
        )

        results.append({
            "model_id": type(model).__name__,
            "track_ids": list(recommendations)
        })

    random.shuffle(results)

    st.session_state.trial = {
        "id": str(uuid4()),
        "input_track_ids": input_tracks_jazz["track_id"].tolist(),
        "results": results
    }



# ------------------------ Show results and Score models ------------------------

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
            #playlist_id = spo.create_playlist(token)
            spo.add_tracks_to_playlist(token, recommendation_playlist_ids[i], result["track_ids"])
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
