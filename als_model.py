"""
als_model.py
============
Alternating Least Squares (ALS) for implicit / explicit feedback.

We implement a pure NumPy ALS for explicit ratings so there are no
external dependencies beyond NumPy/SciPy.

ALS alternately fixes one factor matrix and solves for the other:
    Minimise  Σ_{(u,i) observed} (r_ui - p_u · q_i)²  +  λ(‖P‖² + ‖Q‖²)

Closed-form solution for each user factor:
    p_u = (Q_Iu^T Q_Iu + λI)^{-1} Q_Iu^T r_Iu

where Q_Iu is the sub-matrix of item factors for movies rated by user u,
and r_Iu is the corresponding rating vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ALSModel:
    """
    Explicit-feedback ALS Matrix Factorization.

    Parameters
    ----------
    n_factors   : dimensionality of latent space
    n_iterations: number of ALS alternation steps
    reg         : L2 regularisation weight (lambda)
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 15,
        reg: float = 0.1,
    ) -> None:
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self._P: Optional[np.ndarray] = None  # user factors (n_users × n_factors)
        self._Q: Optional[np.ndarray] = None  # item factors (n_items × n_factors)
        self._user_idx: Dict[int, int] = {}
        self._movie_idx: Dict[int, int] = {}
        self._idx_movie: Dict[int, int] = {}
        self._global_mean: float = 0.0
        self._user_ratings: dict = {}  # {user_inner_idx: [(item_inner_idx, rating)]}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "ALSModel":
        self._global_mean = float(train["rating"].mean())

        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._movie_idx = {m: j for j, m in enumerate(movies)}
        self._idx_movie = {j: m for m, j in self._movie_idx.items()}

        n_users, n_items = len(users), len(movies)
        rng = np.random.default_rng(42)
        self._P = rng.normal(0, 0.01, (n_users, self.n_factors)).astype(np.float32)
        self._Q = rng.normal(0, 0.01, (n_items, self.n_factors)).astype(np.float32)

        # Build lookup structures
        user_ratings: dict[int, list] = {i: [] for i in range(n_users)}
        item_ratings: dict[int, list] = {j: [] for j in range(n_items)}
        for row in train.itertuples():
            u = self._user_idx[row.user_id]
            m = self._movie_idx[row.movie_id]
            r = float(row.rating)
            user_ratings[u].append((m, r))
            item_ratings[m].append((u, r))

        self._user_ratings = user_ratings
        reg_I = self.reg * np.eye(self.n_factors, dtype=np.float32)

        print(f"  ALS: {n_users} users × {n_items} items, {self.n_factors} factors")

        for iteration in range(self.n_iterations):
            # --- Fix Q, solve for P ---
            for u in range(n_users):
                items_rated = user_ratings[u]
                if not items_rated:
                    continue
                idxs = [i for i, _ in items_rated]
                ratings_u = np.array([r for _, r in items_rated], dtype=np.float32)
                Q_u = self._Q[idxs]  # (n_rated × n_factors)
                A = Q_u.T @ Q_u + reg_I
                b = Q_u.T @ ratings_u
                self._P[u] = np.linalg.solve(A, b)

            # --- Fix P, solve for Q ---
            for m in range(n_items):
                users_rated = item_ratings[m]
                if not users_rated:
                    continue
                idxs = [u for u, _ in users_rated]
                ratings_m = np.array([r for _, r in users_rated], dtype=np.float32)
                P_m = self._P[idxs]  # (n_rated_by_item × n_factors)
                A = P_m.T @ P_m + reg_I
                b = P_m.T @ ratings_m
                self._Q[m] = np.linalg.solve(A, b)

            # Training RMSE
            train_preds = np.array([
                float(self._P[self._user_idx[row.user_id]] @ self._Q[self._movie_idx[row.movie_id]])
                if row.user_id in self._user_idx and row.movie_id in self._movie_idx else self._global_mean
                for row in train.itertuples()
            ])
            rmse = float(np.sqrt(np.mean((train["rating"].values - train_preds) ** 2)))
            print(f"  Iter {iteration + 1:2d}/{self.n_iterations}  train RMSE={rmse:.4f}")

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _score(self, user_id: int, movie_id: int) -> float:
        u = self._user_idx.get(user_id)
        m = self._movie_idx.get(movie_id)
        if u is None or m is None:
            return self._global_mean
        return float(self._P[u] @ self._Q[m])

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.array([
            self._score(row.user_id, row.movie_id)
            for row in test.itertuples()
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        u = self._user_idx.get(user_id)
        if u is None or self._P is None or self._Q is None:
            return []
        seen = {m for m, _ in self._user_ratings.get(u, [])}
        scores = self._P[u] @ self._Q.T  # (n_items,)
        ranked = np.argsort(scores)[::-1]
        result = []
        for idx in ranked:
            if idx not in seen:
                result.append(self._idx_movie[int(idx)])
            if len(result) == top_k:
                break
        return result
