"""
user_cf.py
==========
User-Based Collaborative Filtering.

For each target user, we:
1. Compute cosine similarity against all other users.
2. Select the top-K neighbours.
3. Predict ratings as a weighted average of neighbour ratings.
4. Recommend the highest-predicted unseen movies.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserCF:
    """
    User-Based Collaborative Filtering via cosine similarity.

    Parameters
    ----------
    n_neighbours : number of similar users to consider
    min_common   : minimum co-rated items required to be considered a neighbour
    """

    def __init__(self, n_neighbours: int = 50, min_common: int = 5) -> None:
        self.n_neighbours = n_neighbours
        self.min_common = min_common
        self._user_idx: Dict[int, int] = {}
        self._movie_idx: Dict[int, int] = {}
        self._idx_movie: Dict[int, int] = {}
        self._matrix: Optional[np.ndarray] = None
        self._sparse: Optional[csr_matrix] = None
        self._global_mean: float = 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "UserCF":
        """
        Build the user-item matrix from training interactions.
        """
        self._global_mean = float(train["rating"].mean())

        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._movie_idx = {m: j for j, m in enumerate(movies)}
        self._idx_movie = {j: m for m, j in self._movie_idx.items()}

        n_users, n_movies = len(users), len(movies)
        # Dense matrix (fine for sub-500K sample)
        mat = np.zeros((n_users, n_movies), dtype=np.float32)
        for _, row in train.iterrows():
            u = self._user_idx[row["user_id"]]
            m = self._movie_idx[row["movie_id"]]
            mat[u, m] = row["rating"]

        self._matrix = mat
        # Sparse version for efficient similarity
        self._sparse = csr_matrix(mat)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _get_user_vector(self, user_id: int) -> Optional[np.ndarray]:
        idx = self._user_idx.get(user_id)
        if idx is None:
            return None
        return self._matrix[idx]

    def _similar_users(self, user_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (neighbour_indices, similarities) sorted by similarity desc."""
        u_idx = self._user_idx.get(user_id)
        if u_idx is None:
            return np.array([]), np.array([])

        u_vec = self._matrix[u_idx].reshape(1, -1)
        sims = cosine_similarity(u_vec, self._matrix)[0]
        sims[u_idx] = -1  # exclude self

        top_idx = np.argsort(sims)[::-1][: self.n_neighbours]
        return top_idx, sims[top_idx]

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        m_idx = self._movie_idx.get(movie_id)
        if m_idx is None:
            return self._global_mean

        neighbour_idxs, sims = self._similar_users(user_id)
        if len(neighbour_idxs) == 0:
            return self._global_mean

        ratings = self._matrix[neighbour_idxs, m_idx]
        mask = ratings > 0  # only neighbours who rated this movie
        if mask.sum() == 0:
            return self._global_mean

        weighted = np.dot(sims[mask], ratings[mask])
        denom = np.abs(sims[mask]).sum()
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
        if u_idx is None:
            return []

        seen_mask = self._matrix[u_idx] > 0
        neighbour_idxs, sims = self._similar_users(user_id)
        if len(neighbour_idxs) == 0:
            return []

        # Weighted sum of neighbour ratings for unseen movies
        neighbour_ratings = self._matrix[neighbour_idxs]  # (K, movies)
        sim_weights = sims.reshape(-1, 1)
        score = (neighbour_ratings * sim_weights).sum(axis=0)
        norm = np.abs(sim_weights).sum()
        score = score / norm if norm > 0 else score

        # Zero out already-seen movies
        score[seen_mask] = -np.inf

        top_idxs = np.argsort(score)[::-1][:top_k]
        return [self._idx_movie[i] for i in top_idxs if score[i] > -np.inf]
