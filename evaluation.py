"""
evaluation.py
=============
Evaluation metrics for recommendation systems.

Mandatory metrics
-----------------
- RMSE  : Root Mean Squared Error for rating prediction accuracy.
- MAP@10: Mean Average Precision at 10 for ranking quality.
          A movie is considered relevant if its actual rating >= 3.5.

Optional metrics (implemented as extras)
-----------------------------------------
- MAE, Precision@K, Recall@K, NDCG@K, Hit Rate@K, Coverage, Diversity.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Rating prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

RELEVANCE_THRESHOLD = 3.5  # rating >= this is considered "relevant"


def _average_precision_at_k(
    recommended: List[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    """
    Average Precision @ K for a single user.

    Parameters
    ----------
    recommended : ordered list of recommended item IDs (best first)
    relevant    : set of item IDs actually relevant to the user
    k           : cutoff
    """
    if not relevant:
        return 0.0

    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def map_at_k(
    test_df: pd.DataFrame,
    recommend_fn: Callable[[int, int], List[int]],
    k: int = 10,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    verbose: bool = False,
) -> float:
    """
    Mean Average Precision @ K across all users in `test_df`.

    Parameters
    ----------
    test_df          : DataFrame with columns [user_id, movie_id, rating]
    recommend_fn     : callable(user_id, k) -> list of recommended movie_ids
    k                : number of recommendations to consider
    relevance_threshold : rating value above which a movie is "relevant"
    verbose          : print per-user AP (useful for debugging)
    """
    user_ap: list[float] = []

    for user_id, group in test_df.groupby("user_id"):
        relevant_items = set(
            group.loc[group["rating"] >= relevance_threshold, "movie_id"]
        )
        if not relevant_items:
            continue  # skip users with no relevant items in test set

        try:
            recs = recommend_fn(int(user_id), k)
        except Exception:
            recs = []

        ap = _average_precision_at_k(recs, relevant_items, k)
        user_ap.append(ap)

        if verbose:
            print(f"  user={user_id}  AP@{k}={ap:.4f}  relevant={len(relevant_items)}")

    return float(np.mean(user_ap)) if user_ap else 0.0


# ---------------------------------------------------------------------------
# Precision / Recall / Hit-Rate / NDCG
# ---------------------------------------------------------------------------

def precision_at_k(
    recommended: List[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    recs_k = recommended[:k]
    if not recs_k:
        return 0.0
    return len(set(recs_k) & relevant) / k


def recall_at_k(
    recommended: List[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    if not relevant:
        return 0.0
    recs_k = recommended[:k]
    return len(set(recs_k) & relevant) / len(relevant)


def hit_rate_at_k(
    recommended: List[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    return float(bool(set(recommended[:k]) & relevant))


def ndcg_at_k(
    recommended: List[int],
    relevant: set[int],
    k: int = 10,
) -> float:
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# System-level metrics
# ---------------------------------------------------------------------------

def coverage(
    all_recommendations: List[List[int]],
    catalog_size: int,
) -> float:
    """
    Fraction of the item catalog covered across all recommendation lists.
    """
    covered = set(item for recs in all_recommendations for item in recs)
    return len(covered) / catalog_size if catalog_size > 0 else 0.0


def intra_list_diversity(
    recommendations: List[int],
    item_feature_matrix: np.ndarray,
    item_index: Dict[int, int],
) -> float:
    """
    Average pairwise cosine distance within a recommendation list.
    Higher = more diverse recommendations.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    indices = [item_index[i] for i in recommendations if i in item_index]
    if len(indices) < 2:
        return 0.0
    vecs = item_feature_matrix[indices]
    sim_matrix = cosine_similarity(vecs)
    n = len(indices)
    total_sim = (sim_matrix.sum() - n) / (n * (n - 1))  # exclude diagonal
    return float(1 - total_sim)


# ---------------------------------------------------------------------------
# Comprehensive evaluation runner
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    recommend_fn: Callable[[int, int], List[int]],
    k: int = 10,
    max_map_users: int = 1000,
) -> Dict[str, float]:
    """
    Run all mandatory and optional metrics for a model.

    Parameters
    ----------
    model_name    : display name for logging
    y_true        : actual ratings (aligned with y_pred)
    y_pred        : predicted ratings
    test_df       : full test DataFrame for ranking metrics
    recommend_fn  : callable(user_id, k) -> list of movie_ids
    k             : cutoff for ranking metrics
    max_map_users : cap on users evaluated for MAP (speed)
    """
    results: Dict[str, float] = {}

    # Rating prediction
    results["RMSE"] = rmse(y_true, y_pred)
    results["MAE"] = mae(y_true, y_pred)

    # Sampling users for MAP (expensive operation)
    sample_users = test_df["user_id"].unique()
    if len(sample_users) > max_map_users:
        rng = np.random.default_rng(42)
        sample_users = rng.choice(sample_users, size=max_map_users, replace=False)

    sample_test = test_df[test_df["user_id"].isin(sample_users)]
    results[f"MAP@{k}"] = map_at_k(sample_test, recommend_fn, k=k)

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    for metric, value in results.items():
        print(f"  {metric:<12s} {value:.4f}")

    return results


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Render a comparison table from a dict of {model_name: metrics_dict}.
    """
    df = pd.DataFrame(results).T
    df = df[sorted(df.columns)]
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 60 + "\n")
    return df
