"""
recommender.py
==============
High-level interface for generating and analysing Top-K recommendations.

Wraps any fitted model exposing a `.recommend(user_id, top_k)` method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class Recommender:
    """
    Top-K recommendation generator with explanation and analysis utilities.

    Parameters
    ----------
    model  : fitted model with a `.recommend(user_id, top_k)` method
    movies : movie metadata DataFrame (columns: movie_id, title, year)
    ratings: full ratings DataFrame (used to look up user history)
    """

    def __init__(
        self,
        model: Any,
        movies: Optional[pd.DataFrame] = None,
        ratings: Optional[pd.DataFrame] = None,
    ) -> None:
        self.model = model
        self.movies = movies
        self.ratings = ratings

        if movies is not None:
            self._movie_meta: Dict[int, dict] = {
                int(row.movie_id): {"title": row.title, "year": getattr(row, "year", None)}
                for row in movies.itertuples()
            }
        else:
            self._movie_meta = {}

    # ------------------------------------------------------------------
    # Core recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """
        Generate Top-K recommendations for a user.

        Returns a DataFrame with columns:
            rank, movie_id, title, year
        """
        movie_ids = self.model.recommend(user_id, top_k)

        rows = []
        for rank, mid in enumerate(movie_ids, start=1):
            meta = self._movie_meta.get(int(mid), {})
            rows.append({
                "rank": rank,
                "movie_id": int(mid),
                "title": meta.get("title", f"Movie {mid}"),
                "year": meta.get("year"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # User history
    # ------------------------------------------------------------------

    def user_history(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Return the user's most recent ratings."""
        if self.ratings is None:
            return pd.DataFrame()
        user_df = (
            self.ratings[self.ratings["user_id"] == user_id]
            .sort_values("date", ascending=False) if "date" in self.ratings.columns
            else self.ratings[self.ratings["user_id"] == user_id]
        ).head(n)

        if self._movie_meta:
            user_df = user_df.copy()
            user_df["title"] = user_df["movie_id"].map(
                lambda m: self._movie_meta.get(int(m), {}).get("title", str(m))
            )
        return user_df

    # ------------------------------------------------------------------
    # Explainable recommendations (optional task A)
    # ------------------------------------------------------------------

    def explain(self, user_id: int, movie_id: int, top_n_similar: int = 3) -> str:
        """
        Generate a human-readable explanation for why movie_id is recommended.

        For CF/MF models we find movies the user liked that are conceptually
        similar (highly rated by other users who also liked movie_id).
        """
        if self.ratings is None:
            return "Explanation unavailable — no ratings data provided."

        # Movies user rated highly
        user_liked = (
            self.ratings[
                (self.ratings["user_id"] == user_id) & (self.ratings["rating"] >= 4)
            ]["movie_id"]
            .tolist()
        )

        def title(mid: int) -> str:
            return self._movie_meta.get(int(mid), {}).get("title", f"Movie {mid}")

        target_title = title(movie_id)

        if not user_liked:
            return f"'{target_title}' is highly rated by users with similar tastes."

        # Find users who also liked the target movie
        users_liked_target = set(
            self.ratings[
                (self.ratings["movie_id"] == movie_id) & (self.ratings["rating"] >= 4)
            ]["user_id"]
        )

        # What else did those users rate highly?
        if users_liked_target:
            common = (
                self.ratings[
                    self.ratings["user_id"].isin(users_liked_target)
                    & self.ratings["movie_id"].isin(user_liked)
                ]["movie_id"]
                .value_counts()
                .head(top_n_similar)
                .index.tolist()
            )
        else:
            common = user_liked[:top_n_similar]

        if common:
            liked_titles = " and ".join(f"'{title(m)}'" for m in common)
            return (
                f"Because you enjoyed {liked_titles}, "
                f"users with similar taste also loved '{target_title}'."
            )
        return f"'{target_title}' is trending among users with preferences similar to yours."

    # ------------------------------------------------------------------
    # Batch analysis: success & failure cases
    # ------------------------------------------------------------------

    def analyse_recommendations(
        self,
        test_df: pd.DataFrame,
        sample_users: int = 20,
        top_k: int = 10,
        relevance_threshold: float = 3.5,
    ) -> Dict[str, list]:
        """
        Show recommendation success and failure cases for a sample of users.

        Returns dict with keys 'success' and 'failure', each a list of
        (user_id, recommendations_df, hit_count) tuples.
        """
        users = test_df["user_id"].unique()
        rng = np.random.default_rng(42)
        if len(users) > sample_users:
            users = rng.choice(users, size=sample_users, replace=False)

        success, failure = [], []
        for uid in users:
            recs = self.recommend(uid, top_k)
            relevant = set(
                test_df.loc[
                    (test_df["user_id"] == uid) & (test_df["rating"] >= relevance_threshold),
                    "movie_id",
                ]
            )
            hits = len(set(recs["movie_id"]) & relevant)
            entry = (int(uid), recs, hits)
            (success if hits > 0 else failure).append(entry)

        return {"success": success[:5], "failure": failure[:5]}

    def print_recommendations(self, user_id: int, top_k: int = 10) -> None:
        recs = self.recommend(user_id, top_k)
        print(f"\n{'='*55}")
        print(f"  Top-{top_k} Recommendations for User {user_id}")
        print(f"{'='*55}")
        for _, row in recs.iterrows():
            year = f"({int(row['year'])})" if pd.notna(row.get("year")) else ""
            print(f"  {row['rank']:2d}. {row['title']} {year}")
        print(f"{'='*55}\n")
