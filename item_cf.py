"""
item_cf.py
==========
Item-Based Collaborative Filtering.

For each target (user, movie) pair, we:
1. Find movies similar to the target movie (by cosine similarity of rating vectors).
2. Predict rating as a weighted average of user's ratings for similar movies.
3. Rank all unseen movies by predicted rating to generate recommendations.

Item-based CF is generally more scalable than user-based CF because:
- The item similarity matrix can be pre-computed and cached offline.
- Number of items << number of users in most platforms.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemCF:
    """
    Item-Based Collaborative Filtering.

    Parameters
    ----------
    n_neighbours : number of similar items to use for prediction
    """

    def __init__(self, n_neighbours: int = 50) -> None:
        self.n_neighbours = n_neighbours
        self._user_idx: Dict[int, int] = {}
        self._movie_idx: Dict[int, int] = {}
        self._idx_movie: Dict[int, int] = {}
        self._matrix: Optional[np.ndarray] = None       # (users × movies)
        self._item_matrix: Optional[np.ndarray] = None  # (movies × users)
        self._sim: Optional[np.ndarray] = None          # (movies × movies)
        self._global_mean: float = 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "ItemCF":
        self._global_mean = float(train["rating"].mean())

        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._movie_idx = {m: j for j, m in enumerate(movies)}
        self._idx_movie = {j: m for m, j in self._movie_idx.items()}

        n_users, n_movies = len(users), len(movies)
        mat = np.zeros((n_users, n_movies), dtype=np.float32)
        for _, row in train.iterrows():
            u = self._user_idx[row["user_id"]]
            m = self._movie_idx[row["movie_id"]]
            mat[u, m] = row["rating"]

        self._matrix = mat
        self._item_matrix = mat.T  # (movies × users)

        print("  Computing item-item similarity matrix …")
        self._sim = cosine_similarity(self._item_matrix)  # (movies × movies)
        np.fill_diagonal(self._sim, 0)  # exclude self-similarity
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        u_idx = self._user_idx.get(user_id)
        m_idx = self._movie_idx.get(movie_id)

        if u_idx is None or m_idx is None:
            return self._global_mean

        # Find top-K similar movies that this user has rated
        sims = self._sim[m_idx].copy()
        user_rated_mask = self._matrix[u_idx] > 0
        sims[~user_rated_mask] = -1  # ignore unrated

        top_idxs = np.argsort(sims)[::-1][: self.n_neighbours]
        top_sims = sims[top_idxs]
        top_ratings = self._matrix[u_idx][top_idxs]

        valid = top_sims > 0
        if valid.sum() == 0:
            return self._global_mean

        weighted = np.dot(top_sims[valid], top_ratings[valid])
        denom = top_sims[valid].sum()
        return float(weighted / denom) if denom > 0 else self._global_mean

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in test.iterrows():
            preds.append(self.predict_rating(row["user_id"], row["movie_id"]))
        return np.array(preds, dtype=np.float32)

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        u_idx = self._user_idx.get(user_id)
        if u_idx is None or self._sim is None:
            return []

        user_ratings = self._matrix[u_idx]           # (n_movies,)
        seen_mask = user_ratings > 0
        unseen_mask = ~seen_mask

        if not unseen_mask.any():
            return []

        # Score each unseen movie as a weighted sum of similar seen movies
        # (n_unseen × n_seen) @ (n_seen,) → (n_unseen,)
        unseen_idxs = np.where(unseen_mask)[0]
        seen_idxs = np.where(seen_mask)[0]

        sim_block = self._sim[np.ix_(unseen_idxs, seen_idxs)]  # (n_unseen × n_seen)
        rated = user_ratings[seen_idxs]                         # (n_seen,)

        scores = (sim_block * rated).sum(axis=1) / (sim_block.sum(axis=1) + 1e-9)
        top = np.argsort(scores)[::-1][:top_k]
        return [self._idx_movie[unseen_idxs[i]] for i in top]
