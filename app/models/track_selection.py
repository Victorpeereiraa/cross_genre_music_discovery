from pathlib import Path
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

from app.models.abstract_model import AbstractModel


class TrackSelection:

    def __init__(self, catalog=None, max_suggestions=10, min_chars=2):

        self.max_tracks = AbstractModel.input_size
        self.max_suggestions = max_suggestions
        self.min_chars = min_chars

        # Loading catalog (needs track_name and artists; genres is optional)
        if catalog is None:
            root = Path(__file__).resolve().parents[2]
            catalog = pd.read_parquet(root / "artifacts" / "catalog_clean.parquet")

        if "track_id" in catalog.columns:
            catalog = catalog.set_index("track_id")

        missing = [c for c in ("track_name", "artists") if c not in catalog.columns]
        if missing:
            raise ValueError(f"The catalog needs the columns: {missing}")

        self.catalog_clean = catalog[~catalog.index.duplicated()]

        # Search columns live in their own dataframe (the catalog is not changed)
        genre_col = next(
            (c for c in self.catalog_clean.columns if c.strip().lower() in ("genres", "genre", "track_genre")),
            None
        )

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

        self.fields = {"Tudo": "all", "Música": "track_name", "Artista": "artists"}
        if genre_col:
            self.fields["Gênero"] = "genres"

        self.labels = dict(zip(
            self.catalog_clean.index,
            self.catalog_clean["track_name"].astype(str) + " — " + self.catalog_clean["artists"].astype(str)
        ))

        self.selected = []        # selected track ids, in order
        self.updating = False     # avoids triggering the click when the list is redrawn

        self._build_ui()


    @staticmethod
    def _to_text(value):

        # genres can be a list, an array or a string
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set, np.ndarray)):
            return " ".join(map(str, value))
        return "" if pd.isna(value) else str(value)


    def _build_ui(self):

        self.field_dropdown = widgets.Dropdown(options=list(self.fields), value="Tudo", description="Campo:")
        self.search_box = widgets.Text(
            placeholder="Digite uma música, artista ou gênero...",
            description="Buscar:",
            layout=widgets.Layout(width="500px")
        )
        self.suggestions = widgets.Select(options=[], rows=8, layout=widgets.Layout(width="500px"))
        self.counter = widgets.HTML()
        self.chosen = widgets.Select(options=[], rows=self.max_tracks, layout=widgets.Layout(width="500px"))
        self.remove_button = widgets.Button(description="Remover", icon="trash")
        self.clear_button = widgets.Button(description="Limpar tudo")

        self.search_box.observe(self._update_suggestions, names="value")
        self.field_dropdown.observe(self._update_suggestions, names="value")
        self.suggestions.observe(self._add, names="value")
        self.remove_button.on_click(self._remove)
        self.clear_button.on_click(self._clear)

        self.ui = widgets.VBox([
            self.field_dropdown,
            self.search_box,
            self.suggestions,
            self.counter,
            self.chosen,
            widgets.HBox([self.remove_button, self.clear_button])
        ])

        self._redraw()


    def _redraw(self, warning=""):

        self.chosen.options = [(self.labels[t], t) for t in self.selected]

        status = " ✅ pronto!" if self.is_complete else ""
        self.counter.value = f"<b>Selecionadas: {len(self.selected)}/{self.max_tracks}</b>{status} {warning}"


    def _update_suggestions(self, change=None):

        term = self.search_box.value.strip().lower()

        self.updating = True

        if len(term) < self.min_chars:
            self.suggestions.options = []
        else:
            mask = self.search[self.fields[self.field_dropdown.value]].str.contains(term, regex=False)
            ids = self.search.index[mask][:self.max_suggestions]

            self.suggestions.options = [
                (("✓ " if t in self.selected else "") + self.labels[t], t) for t in ids
            ]
            self.suggestions.index = None

        self.updating = False


    def _add(self, change):

        track_id = change["new"]

        if self.updating or track_id is None:
            return

        warning = ""

        if len(self.selected) >= self.max_tracks:
            warning = "⚠️ limite atingido, remova uma para adicionar outra."
        elif track_id in self.selected:
            warning = "⚠️ essa música já está na lista."
        else:
            self.selected.append(track_id)

        # lets the user click the same suggestion again
        self.updating = True
        self.suggestions.index = None
        self.updating = False

        self._redraw(warning)
        self._update_suggestions()


    def _remove(self, _):

        if self.chosen.value in self.selected:
            self.selected.remove(self.chosen.value)

        self._redraw()
        self._update_suggestions()


    def _clear(self, _):

        self.selected.clear()

        self._redraw()
        self._update_suggestions()


    def show(self):

        display(self.ui)


    @property
    def track_ids(self):

        return list(self.selected)


    @property
    def is_complete(self):

        return len(self.selected) == self.max_tracks


    @property
    def selected_tracks(self):

        # selected tracks, in the order they were chosen
        return self.catalog_clean.loc[self.selected].reset_index()


    def clear(self):

        self._clear(None)


    def validate(self, track_ids=None):

        track_ids = self.track_ids if track_ids is None else track_ids

        if not isinstance(track_ids, list):
            raise TypeError("The input must be a list of track ids")

        if len(track_ids) != self.max_tracks:
            raise ValueError(f"The list must have {self.max_tracks} tracks (received: {len(track_ids)})")

        if len(set(track_ids)) != len(track_ids):
            raise ValueError("The list has repeated tracks")

        not_found = [t for t in track_ids if t not in self.catalog_clean.index]
        if not_found:
            raise ValueError(f"Tracks not found in the catalog: {not_found}")

        return track_ids
