
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from app.models.abstract_model import AbstractModel


class TrackSelection:

    def __init__(self, catalog=None, max_suggestions=10, min_chars=2, key="track_selection"):

        self.max_tracks = AbstractModel.input_size
        self.max_suggestions = max_suggestions
        self.min_chars = min_chars
        self.key = key

        # Load catalog
        if catalog is None:
            root = Path(__file__).resolve().parents[2]
            catalog = pd.read_parquet(root / "artifacts" / "catalog_clean.parquet")

        if "track_id" in catalog.columns:
            catalog = catalog.set_index("track_id")

        missing = [c for c in ("track_name", "artists") if c not in catalog.columns]
        if missing:
            raise ValueError(f"The catalog needs the columns: {missing}")

        self.catalog_clean = catalog[~catalog.index.duplicated()].copy()

        # Identify genre column
        genre_col = next(
            (
                c for c in self.catalog_clean.columns
                if str(c).strip().lower() in ("genres", "genre", "track_genre")
            ),
            None
        )

        # Prepare search dataframe
        track_name = self.catalog_clean["track_name"].fillna("").astype(str).str.lower()
        artists = self.catalog_clean["artists"].fillna("").astype(str).str.lower()

        genres = (
            self.catalog_clean[genre_col].apply(self._to_text).str.lower()
            if genre_col else pd.Series("", index=self.catalog_clean.index)
        )

        self.search = pd.DataFrame({
            "all": track_name + " " + artists + " " + genres,
            "track_name": track_name,
            "artists": artists,
            "genres": genres
        })

        self.fields = {
            "All": "all",
            "Song name": "track_name",
            "Artist": "artists"
        }

        if genre_col:
            self.fields["Genre"] = "genres"

        # Labels displayed to the user
        self.labels = dict(zip(
            self.catalog_clean.index,
            self.catalog_clean["track_name"].astype(str)
            + " — "
            + self.catalog_clean["artists"].astype(str)
        ))

        # Persistent selection
        if f"{self.key}_selected" not in st.session_state:
            st.session_state[f"{self.key}_selected"] = []


    @staticmethod
    def _to_text(value):

        if isinstance(value, str):
            return value

        if isinstance(value, (list, tuple, set, np.ndarray)):
            return " ".join(map(str, value))

        return "" if pd.isna(value) else str(value)


    @property
    def track_ids(self):

        return list(st.session_state[f"{self.key}_selected"])


    @property
    def is_complete(self):

        return len(self.track_ids) == self.max_tracks


    @property
    def selected_tracks(self):

        return self.catalog_clean.loc[self.track_ids].reset_index()


    def add(self, track_id):

        selected = st.session_state[f"{self.key}_selected"]

        if track_id in selected:
            return False

        if len(selected) >= self.max_tracks:
            return False

        selected.append(track_id)
        return True


    def remove(self, track_id):

        selected = st.session_state[f"{self.key}_selected"]

        if track_id in selected:
            selected.remove(track_id)


    def clear(self):

        st.session_state[f"{self.key}_selected"] = []


    def validate(self, track_ids=None):

        track_ids = self.track_ids if track_ids is None else track_ids

        if not isinstance(track_ids, list):
            raise TypeError("The input must be a list of track ids")

        if len(track_ids) != self.max_tracks:
            raise ValueError(
                f"The list must have {self.max_tracks} tracks "
                f"(received: {len(track_ids)})"
            )

        if len(set(track_ids)) != len(track_ids):
            raise ValueError("The list has repeated tracks")

        not_found = [
            t for t in track_ids
            if t not in self.catalog_clean.index
        ]

        if not_found:
            raise ValueError(f"Tracks not found in the catalog: {not_found}")

        return track_ids


    def render(self):

        #st.subheader("Escolha suas músicas")

        st.markdown(
            f"# Select {self.max_tracks} tracks"
        )

        # Search field
        selected_field = st.selectbox(
            "Search by:",
            options=list(self.fields.keys()),
            key=f"{self.key}_field"
        )

        query = st.text_input(
            "Search by song, artist or genre:",
            placeholder="Enter at least 2 characters...",
            key=f"{self.key}_query"
        )

        # Search catalog
        suggestions = []

        if len(query.strip()) >= self.min_chars:

            column = self.fields[selected_field]

            mask = self.search[column].str.contains(
                query.strip().lower(),
                regex=False
            )

            suggestions = [
                track_id
                for track_id in self.search.index[mask]
                if track_id not in self.track_ids
            ][:self.max_suggestions]

        # Display search results
        if suggestions:

            selected_suggestion = st.selectbox(
                "Result found:",
                options=suggestions,
                format_func=lambda track_id: self.labels[track_id],
                key=f"{self.key}_suggestion"
            )

            if st.button(
                "Add song",
                disabled=self.is_complete,
                key=f"{self.key}_add"
            ):

                self.add(selected_suggestion)
                st.rerun()

        elif len(query.strip()) >= self.min_chars:

            st.info("No songs found.")

        # Display selected tracks
        st.divider()

        st.write(
            f"**Songs selected: {len(self.track_ids)}/{self.max_tracks}**"
        )

        for i, track_id in enumerate(self.track_ids):

            col_name, col_remove = st.columns([5, 1])

            col_name.write(self.labels[track_id])

            if col_remove.button(
                "Remove song",
                key=f"{self.key}_remove_{i}"
            ):

                self.remove(track_id)
                st.rerun()

        # Clear selection
        if self.track_ids:

            if st.button(
                "Clear selection",
                key=f"{self.key}_clear"
            ):

                self.clear()
                st.rerun()

        st.divider()

        # Recommendation button
        recommend_clicked = st.button(
            "Add songs to playlist",
            type="primary",
            disabled=not self.is_complete,
            key=f"{self.key}_recommend"
        )

        if recommend_clicked:
            return self.validate()

        return None
