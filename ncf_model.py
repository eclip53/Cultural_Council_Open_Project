"""
ncf_model.py
============
Neural Collaborative Filtering (NCF) — He et al. (2017).

Architecture
------------
NCF fuses two pathways:
  1. GMF (Generalised Matrix Factorisation): element-wise product of
     user and item embeddings — captures linear interactions.
  2. MLP pathway: concatenated embeddings passed through dense layers
     — captures non-linear interactions.

The outputs of both pathways are concatenated and fed to a final sigmoid
(or linear for explicit ratings) output layer.

Reference:
    He, X. et al. (2017). Neural Collaborative Filtering. WWW '17.
    https://arxiv.org/abs/1708.05031
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. NCFModel will be unavailable.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RatingDataset(Dataset):
    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _NCFNet(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_factors: int = 32,
        mlp_factors: int = 32,
        mlp_layers: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, gmf_factors)
        self.gmf_item_emb = nn.Embedding(n_items, gmf_factors)

        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(n_users, mlp_factors)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_factors)

        # MLP tower
        mlp_input_dim = mlp_factors * 2
        layers = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Final prediction layer (output = predicted rating 1-5)
        self.output = nn.Linear(gmf_factors + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF path
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i  # element-wise

        # MLP path
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))

        # Fusion & prediction
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        pred = self.output(fused).squeeze(-1)
        # Clamp to rating range [1, 5]
        return torch.clamp(pred, 1.0, 5.0)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class NCFModel:
    """
    Neural Collaborative Filtering model.

    Parameters
    ----------
    gmf_factors : embedding size for GMF path
    mlp_factors : embedding size for MLP path
    mlp_layers  : hidden dimensions of MLP tower
    lr          : learning rate
    batch_size  : mini-batch size
    n_epochs    : training epochs
    dropout     : dropout probability
    device      : 'cuda', 'mps', or 'cpu' (auto-selected if None)
    """

    def __init__(
        self,
        gmf_factors: int = 32,
        mlp_factors: int = 32,
        mlp_layers: Optional[list] = None,
        lr: float = 1e-3,
        batch_size: int = 1024,
        n_epochs: int = 10,
        dropout: float = 0.2,
        device: Optional[str] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("Install PyTorch: pip install torch")

        self.gmf_factors = gmf_factors
        self.mlp_factors = mlp_factors
        self.mlp_layers = mlp_layers or [64, 32, 16]
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout

        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self._net: Optional[_NCFNet] = None
        self._user_idx: Dict[int, int] = {}
        self._movie_idx: Dict[int, int] = {}
        self._idx_movie: Dict[int, int] = {}
        self._global_mean: float = 0.0
        self._all_item_indices: np.ndarray = np.array([])
        self._user_seen: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "NCFModel":
        self._global_mean = float(train["rating"].mean())

        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._movie_idx = {m: j for j, m in enumerate(movies)}
        self._idx_movie = {j: m for m, j in self._movie_idx.items()}
        self._all_item_indices = np.arange(len(movies))

        # Record seen items per user
        for row in train.itertuples():
            u_inner = self._user_idx[row.user_id]
            m_inner = self._movie_idx[row.movie_id]
            self._user_seen.setdefault(u_inner, set()).add(m_inner)

        # Build dataset
        u_arr = np.array([self._user_idx[u] for u in train["user_id"]])
        m_arr = np.array([self._movie_idx[m] for m in train["movie_id"]])
        r_arr = train["rating"].values.astype(np.float32)

        dataset = RatingDataset(u_arr, m_arr, r_arr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self._net = _NCFNet(
            n_users=len(users),
            n_items=len(movies),
            gmf_factors=self.gmf_factors,
            mlp_factors=self.mlp_factors,
            mlp_layers=self.mlp_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        print(f"  NCF on {self.device} | {len(users)} users × {len(movies)} items")

        self._net.train()
        for epoch in range(self.n_epochs):
            t0 = time.time()
            total_loss = 0.0
            for users_b, items_b, ratings_b in loader:
                users_b = users_b.to(self.device)
                items_b = items_b.to(self.device)
                ratings_b = ratings_b.to(self.device)

                optimiser.zero_grad()
                preds = self._net(users_b, items_b)
                loss = criterion(preds, ratings_b)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(ratings_b)

            rmse = float(np.sqrt(total_loss / len(dataset)))
            print(f"  Epoch {epoch + 1:2d}/{self.n_epochs}  loss(MSE)={total_loss/len(dataset):.4f}  RMSE={rmse:.4f}  [{time.time()-t0:.1f}s]")

        self._net.eval()
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _safe_inner(self, user_id: int, movie_id: int) -> tuple[int, int] | None:
        u = self._user_idx.get(user_id)
        m = self._movie_idx.get(movie_id)
        if u is None or m is None:
            return None
        return u, m

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() first.")

        valid_rows = []
        indices = []
        for idx, row in enumerate(test.itertuples()):
            pair = self._safe_inner(row.user_id, row.movie_id)
            if pair is not None:
                valid_rows.append(pair)
                indices.append(idx)

        preds = np.full(len(test), self._global_mean, dtype=np.float32)
        if not valid_rows:
            return preds

        u_arr = torch.LongTensor([p[0] for p in valid_rows]).to(self.device)
        m_arr = torch.LongTensor([p[1] for p in valid_rows]).to(self.device)

        with torch.no_grad():
            out = self._net(u_arr, m_arr).cpu().numpy()

        for i, pred in zip(indices, out):
            preds[i] = float(pred)

        return preds

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        if self._net is None:
            return []
        u = self._user_idx.get(user_id)
        if u is None:
            return []

        seen = self._user_seen.get(u, set())
        unseen = np.array([i for i in self._all_item_indices if i not in seen])
        if len(unseen) == 0:
            return []

        u_tensor = torch.LongTensor([u] * len(unseen)).to(self.device)
        i_tensor = torch.LongTensor(unseen).to(self.device)

        with torch.no_grad():
            scores = self._net(u_tensor, i_tensor).cpu().numpy()

        top = np.argsort(scores)[::-1][:top_k]
        return [self._idx_movie[int(unseen[i])] for i in top]
