"""
svd_model.py
============
Matrix Factorization via SVD using the `scikit-surprise` library.

SVD decomposes the user-item rating matrix R into:
    R ≈ U · Σ · Vᵀ
where U captures user latent factors and V captures item latent factors.
Surprise's SVD adds bias terms and uses SGD optimisation.

Reference:
    Koren, Y. (2008). Factorization meets the neighborhood. KDD '08.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("[WARNING] scikit-surprise not installed. SVDModel will be unavailable.")


class SVDModel:
    """
    Wrapper around Surprise's SVD for the Netflix Prize dataset.

    Parameters
    ----------
    n_factors  : number of latent factors
    n_epochs   : number of SGD epochs
    lr_all     : learning rate
    reg_all    : regularisation term
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ) -> None:
        if not SURPRISE_AVAILABLE:
            raise ImportError("Install scikit-surprise: pip install scikit-surprise")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self._model: Optional[SVD] = None
        self._trainset = None
        self._all_movie_ids: list[int] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "SVDModel":
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train[["user_id", "movie_id", "rating"]], reader
        )
        self._trainset = data.build_full_trainset()
        self._all_movie_ids = sorted(train["movie_id"].unique().tolist())

        self._model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=True,
        )
        self._model.fit(self._trainset)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        pred = self._model.predict(str(user_id), str(movie_id))
        return float(pred.est)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        preds = [
            self._model.predict(str(row.user_id), str(row.movie_id)).est
            for row in test.itertuples()
        ]
        return np.array(preds, dtype=np.float32)

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        if self._model is None or self._trainset is None:
            return []

        # Movies already rated by this user
        try:
            inner_uid = self._trainset.to_inner_uid(str(user_id))
            seen = {
                self._trainset.to_raw_iid(iid)
                for iid, _ in self._trainset.ur[inner_uid]
            }
        except ValueError:
            seen = set()

        candidates = [m for m in self._all_movie_ids if str(m) not in seen]
        scored = [
            (m, self._model.predict(str(user_id), str(m)).est)
            for m in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]
