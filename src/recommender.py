"""Content-based Steam game recommendations using TF-IDF and cosine similarity."""

from __future__ import annotations

import ast
import heapq
import math
import re
from collections import Counter
from typing import Iterable

import pandas as pd


TITLE_COLUMNS = ("name", "title", "game", "game_name", "app_name")
TEXT_COLUMNS = (
    "genres", "genre", "tags", "categories", "description", "about_the_game",
    "short_description", "developer", "developers", "publisher", "publishers",
)
STOP_WORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "you", "your", "are",
    "can", "has", "have", "was", "were", "will", "into", "its", "not", "but",
    "all", "their", "they", "our", "out", "get", "one", "more", "than", "when", "how",
})


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    return next((by_normalized[key] for candidate in candidates
                 if (key := re.sub(r"[^a-z0-9]", "", candidate.lower())) in by_normalized), None)


def _metadata_words(value: object, column: str) -> str:
    """Turn Steam's stringified lists and tag dictionaries into readable tokens."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if column.lower() in {"genres", "genre", "categories", "tags"} and text[:1] in "[{":
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return " ".join(map(str, parsed.keys()))
            if isinstance(parsed, (list, tuple, set)):
                return " ".join(map(str, parsed))
        except (ValueError, SyntaxError):
            # Keep the original text if another dataset uses a non-Python format.
            pass
    # Long store-page copy adds a lot of noise and makes the first model build slower.
    if column.lower() in {"description", "about_the_game", "short_description"}:
        text = text[:320]
    return text


def _metadata_items(value: object) -> list[str]:
    """Read a Steam list field such as ['Action', 'RPG'] into category labels."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text) if text[:1] == "[" else []
    return quoted or [part.strip() for part in re.split(r"[,;]", text) if part.strip()]


def prepare_games(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize Steam catalog fields and build a compact profile per game."""
    if raw.empty:
        raise ValueError("The games file is empty.")
    title_column = _find_column(raw.columns, TITLE_COLUMNS)
    if title_column is None:
        raise ValueError("Couldn't find a game title column. Add a column named Name or Title.")

    games = raw.copy()
    games["_title"] = games[title_column].fillna("").astype(str).str.strip()
    games = games[games["_title"].ne("")].drop_duplicates("_title").reset_index(drop=True)
    usable_text = [column for column in games.columns
                   if re.sub(r"[^a-z0-9]", "", str(column).lower()) in
                   {re.sub(r"[^a-z0-9]", "", name) for name in TEXT_COLUMNS}]
    if not usable_text:
        raise ValueError("The CSV needs game information such as genres, tags, or descriptions.")
    genre_column = _find_column(games.columns, ("genres", "genre"))
    games["_genre_values"] = games[genre_column].map(_metadata_items) if genre_column else [[] for _ in range(len(games))]

    parts = []
    for column in usable_text:
        parts.append(games[column].map(lambda value: _metadata_words(value, str(column))))
    profile = parts[0]
    for part in parts[1:]:
        profile = profile.str.cat(part, sep=" ")
    games["_profile"] = (profile.str.lower()
                         .str.replace(r"[^a-z0-9 ]", " ", regex=True)
                         .str.replace(r"\s+", " ", regex=True).str.strip())
    games = games[games["_profile"].ne("")].reset_index(drop=True)
    return games, str(title_column)


class GameRecommender:
    """Fit a compact TF-IDF model once and rank matches by cosine similarity."""

    def __init__(self, raw: pd.DataFrame, catalog: pd.DataFrame | None = None):
        self.games, self.title_column = prepare_games(raw)
        self.catalog = catalog if catalog is not None else raw
        self.catalog_title_column = _find_column(self.catalog.columns, TITLE_COLUMNS) or self.title_column
        document_frequency: Counter[str] = Counter()
        for profile in self.games["_profile"]:
            document_frequency.update(self._term_counts(profile).keys())

        total = len(self.games)
        lower_bound = 2
        upper_bound = max(2, math.floor(total * 0.92))
        common_terms = [(term, frequency) for term, frequency in document_frequency.items()
                        if lower_bound <= frequency <= upper_bound]
        if len(common_terms) > 40000:
            common_terms = heapq.nlargest(40000, common_terms, key=lambda item: item[1])
        self.idf = {term: math.log((total + 1) / (frequency + 1)) + 1
                    for term, frequency in common_terms}

        self.vectors: list[dict[str, float]] = []
        for profile in self.games["_profile"]:
            terms = self._term_counts(profile)
            weighted = {term: (1 + math.log(count)) * self.idf[term]
                        for term, count in terms.items() if term in self.idf}
            norm = math.sqrt(sum(weight * weight for weight in weighted.values())) or 1.0
            self.vectors.append({term: weight / norm for term, weight in weighted.items()})
        self._title_lookup = {title.casefold(): i for i, title in enumerate(self.games["_title"])}
        self._catalog_lookup = {
            str(title).strip().casefold(): i
            for i, title in enumerate(self.catalog[self.catalog_title_column].fillna(""))
            if str(title).strip()
        }

    @staticmethod
    def _term_counts(profile: str) -> Counter[str]:
        words = re.findall(r"[a-z0-9]{2,}", profile.lower())
        words = [word for word in words if word not in STOP_WORDS]
        tokens = words + [f"{left} {right}" for left, right in zip(words, words[1:])]
        return Counter(tokens)

    @staticmethod
    def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(term, 0.0) for term, weight in left.items())

    def _vectorize(self, profile: str) -> dict[str, float]:
        terms = self._term_counts(profile)
        weighted = {term: (1 + math.log(count)) * self.idf[term]
                    for term, count in terms.items() if term in self.idf}
        norm = math.sqrt(sum(weight * weight for weight in weighted.values())) or 1.0
        return {term: weight / norm for term, weight in weighted.items()}

    def recommend(self, title: str, limit: int = 6) -> pd.DataFrame:
        if len(self.games) < 2:
            return self.games.iloc[0:0].copy()
        key = title.casefold()
        index = self._title_lookup.get(key)
        if index is not None:
            query = self.vectors[index]
        else:
            catalog_index = self._catalog_lookup.get(key)
            if catalog_index is None:
                matches = self.catalog[self.catalog[self.catalog_title_column].astype(str).str.contains(
                    re.escape(title), case=False, na=False
                )]
                if matches.empty:
                    return self.games.iloc[0:0].copy()
                source = matches.iloc[0]
            else:
                source = self.catalog.iloc[catalog_index]
            query_frame, _ = prepare_games(source.to_frame().T)
            if query_frame.empty:
                return self.games.iloc[0:0].copy()
            query = self._vectorize(query_frame.iloc[0]["_profile"])
        top = heapq.nlargest(limit, ((i, self._similarity(query, vector))
                                     for i, vector in enumerate(self.vectors) if i != index),
                             key=lambda item: item[1])
        indices, scores = zip(*top) if top else ((), ())
        results = self.games.iloc[list(indices)].copy()
        results["_match"] = [round(score * 100) for score in scores]
        return results

    def recommend_by_genre(self, genre: str, limit: int = 12) -> pd.DataFrame:
        """Rank games within a selected genre against the genre's average TF-IDF profile."""
        if not genre or "_genre_values" not in self.games.columns:
            return self.games.iloc[0:0].copy()
        mask = self.games["_genre_values"].map(
            lambda values: any(value.casefold() == genre.casefold() for value in values)
        ).to_numpy()
        indices = mask.nonzero()[0]
        if not len(indices):
            return self.games.iloc[0:0].copy()
        center: dict[str, float] = Counter()
        for index in indices:
            center.update(self.vectors[index])
        center_norm = math.sqrt(sum(weight * weight for weight in center.values())) or 1.0
        center_vector = {term: weight / center_norm for term, weight in center.items()}
        popularity = pd.to_numeric(self.games.get("recommendations", 0), errors="coerce")
        if not isinstance(popularity, pd.Series):
            popularity = pd.Series(0, index=self.games.index)
        log_popularity = popularity.fillna(0).clip(lower=0).map(math.log1p)
        max_popularity = float(log_popularity.max()) or 1.0

        def ranked(index: int) -> tuple[int, float, float]:
            similarity = self._similarity(self.vectors[index], center_vector)
            audience_signal = float(log_popularity.iloc[index]) / max_popularity
            return index, similarity, similarity * 0.85 + audience_signal * 0.15

        top = heapq.nlargest(limit, (ranked(int(index)) for index in indices), key=lambda item: item[2])
        picked_indices = tuple(item[0] for item in top)
        scores = tuple(item[1] for item in top)
        results = self.games.iloc[list(picked_indices)].copy()
        results["_match"] = [round(score * 100) for score in scores]
        return results


def demo_games() -> pd.DataFrame:
    """Small offline catalog used only if the Steam CSV is not present."""
    rows = [
        ("Hades", "['Action', 'Indie', 'RPG']", "God-like powers and fast combat in the underworld.", ""),
        ("Dead Cells", "['Action', 'Indie']", "A rogue-lite, metroidvania-inspired action platformer.", ""),
        ("Hollow Knight", "['Action', 'Adventure', 'Indie']", "Forge your own path through a vast ruined kingdom.", ""),
        ("Celeste", "['Adventure', 'Indie']", "Help Madeline survive her journey to the top of Celeste Mountain.", ""),
        ("Stardew Valley", "['Casual', 'Indie', 'RPG', 'Simulation']", "Build a life on the land and make the valley your own.", ""),
        ("Terraria", "['Action', 'Adventure', 'Indie']", "Dig, fight, explore, and build in a world of your own.", ""),
        ("The Witcher 3: Wild Hunt", "['RPG']", "Become Geralt of Rivia, a monster slayer for hire.", ""),
        ("Elden Ring", "['Action', 'RPG']", "Rise, Tarnished, and explore the Lands Between.", ""),
        ("Baldur's Gate 3", "['Adventure', 'RPG', 'Strategy']", "Gather your party in a story-rich adventure of choice and consequence.", ""),
        ("Portal 2", "['Adventure']", "A mind-bending adventure through the Aperture Science labs.", ""),
        ("Counter-Strike 2", "['Action', 'Free To Play']", "A precise, team-based competitive shooter.", ""),
        ("Subnautica", "['Adventure', 'Indie']", "Explore an alien ocean world filled with wonder and danger.", ""),
        ("Slay the Spire", "['Indie', 'Strategy']", "Craft a deck, meet strange creatures, and climb the Spire.", ""),
        ("Forza Horizon 5", "['Racing']", "Explore the vibrant landscapes of Mexico behind the wheel.", ""),
        ("It Takes Two", "['Action', 'Adventure']", "Team up for a genre-bending platform adventure built for two.", ""),
        ("The Sims 4", "['Casual', 'Free To Play', 'Simulation']", "Create Sims and tell stories in a world of your own design.", ""),
    ]
    return pd.DataFrame(rows, columns=["name", "genres", "short_description", "header_image"])
