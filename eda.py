"""
eda.py
======
Exploratory Data Analysis for the Netflix Prize dataset.

All public functions accept a ratings DataFrame and an optional movies
DataFrame and return matplotlib Figure objects (so callers can save or
display them as needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" / "eda"
OUTPUTS.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = sns.color_palette("mako", 8)


# ---------------------------------------------------------------------------
# 1. Basic statistics
# ---------------------------------------------------------------------------

def dataset_summary(ratings: pd.DataFrame, movies: Optional[pd.DataFrame] = None) -> dict:
    """Return a dict of key dataset statistics."""
    stats = {
        "total_ratings": len(ratings),
        "unique_users": ratings["user_id"].nunique(),
        "unique_movies": ratings["movie_id"].nunique(),
        "rating_min": int(ratings["rating"].min()),
        "rating_max": int(ratings["rating"].max()),
        "rating_mean": round(float(ratings["rating"].mean()), 4),
        "rating_std": round(float(ratings["rating"].std()), 4),
        "sparsity": 1 - len(ratings) / (ratings["user_id"].nunique() * ratings["movie_id"].nunique()),
        "date_min": str(ratings["date"].min().date()) if "date" in ratings else "N/A",
        "date_max": str(ratings["date"].max().date()) if "date" in ratings else "N/A",
    }
    if movies is not None:
        stats["total_movies_in_catalog"] = len(movies)
    return stats


def print_summary(ratings: pd.DataFrame, movies: Optional[pd.DataFrame] = None) -> None:
    stats = dataset_summary(ratings, movies)
    print("\n" + "=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"  {label:<30s} {v:.4f}")
        else:
            print(f"  {label:<30s} {v:,}" if isinstance(v, int) else f"  {label:<30s} {v}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# 2. Rating distribution
# ---------------------------------------------------------------------------

def plot_rating_distribution(ratings: pd.DataFrame, save: bool = True) -> plt.Figure:
    counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[2], edgecolor="white", linewidth=0.8)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{bar.get_height():,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Ratings")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUTS / "rating_distribution.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 3. User activity
# ---------------------------------------------------------------------------

def plot_user_activity(ratings: pd.DataFrame, save: bool = True) -> plt.Figure:
    user_counts = ratings.groupby("user_id").size()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Log-scale histogram
    axes[0].hist(user_counts, bins=80, color=PALETTE[3], edgecolor="white", log=True)
    axes[0].set_title("Ratings per User (log scale)", fontweight="bold")
    axes[0].set_xlabel("Number of Ratings")
    axes[0].set_ylabel("Number of Users (log)")

    # Cumulative distribution
    sorted_counts = np.sort(user_counts)[::-1]
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
    axes[1].plot(cumulative, color=PALETTE[1], linewidth=2)
    axes[1].axhline(0.8, color="salmon", linestyle="--", label="80% of ratings")
    axes[1].set_title("Cumulative Ratings by User Rank", fontweight="bold")
    axes[1].set_xlabel("User Rank")
    axes[1].set_ylabel("Cumulative Fraction of All Ratings")
    axes[1].legend()

    plt.suptitle("User Activity Patterns", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUTS / "user_activity.png", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Content popularity
# ---------------------------------------------------------------------------

def plot_content_popularity(
    ratings: pd.DataFrame,
    movies: Optional[pd.DataFrame] = None,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    movie_counts = ratings.groupby("movie_id").size().reset_index(name="n_ratings")
    movie_avg = ratings.groupby("movie_id")["rating"].mean().reset_index(name="avg_rating")
    movie_stats = movie_counts.merge(movie_avg, on="movie_id")

    if movies is not None:
        movie_stats = movie_stats.merge(movies[["movie_id", "title"]], on="movie_id", how="left")
        movie_stats["label"] = movie_stats["title"].fillna(movie_stats["movie_id"].astype(str))
    else:
        movie_stats["label"] = movie_stats["movie_id"].astype(str)

    top = movie_stats.nlargest(top_n, "n_ratings")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top movies by rating count
    axes[0].barh(top["label"][::-1], top["n_ratings"][::-1], color=PALETTE[4])
    axes[0].set_title(f"Top {top_n} Most-Rated Movies", fontweight="bold")
    axes[0].set_xlabel("Number of Ratings")
    axes[0].tick_params(axis="y", labelsize=8)

    # Rating count vs average rating scatter
    axes[1].scatter(
        movie_stats["n_ratings"],
        movie_stats["avg_rating"],
        alpha=0.3,
        s=8,
        color=PALETTE[5],
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Popularity vs Average Rating", fontweight="bold")
    axes[1].set_xlabel("Number of Ratings (log)")
    axes[1].set_ylabel("Average Rating")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUTS / "content_popularity.png", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Temporal trends
# ---------------------------------------------------------------------------

def plot_temporal_trends(ratings: pd.DataFrame, save: bool = True) -> plt.Figure:
    if "date" not in ratings or ratings["date"].isna().all():
        print("No date column available — skipping temporal analysis.")
        return None

    monthly = (
        ratings.set_index("date")
        .resample("ME")["rating"]
        .agg(["count", "mean"])
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].fill_between(monthly["date"], monthly["count"], alpha=0.7, color=PALETTE[2])
    axes[0].set_title("Monthly Rating Volume", fontweight="bold")
    axes[0].set_ylabel("Number of Ratings")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x < 1e6 else f"{x/1e6:.1f}M")
    )

    axes[1].plot(monthly["date"], monthly["mean"], color=PALETTE[5], linewidth=2)
    axes[1].fill_between(monthly["date"], monthly["mean"], alpha=0.3, color=PALETTE[5])
    axes[1].set_title("Monthly Average Rating", fontweight="bold")
    axes[1].set_ylabel("Average Rating")
    axes[1].set_ylim(1, 5)

    plt.suptitle("Temporal Rating Trends", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUTS / "temporal_trends.png", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Sparsity visualisation
# ---------------------------------------------------------------------------

def plot_sparsity_heatmap(
    ratings: pd.DataFrame,
    n_users: int = 200,
    n_movies: int = 200,
    save: bool = True,
) -> plt.Figure:
    """
    Visualise the user-item matrix sparsity on a random sub-sample.
    """
    sample_users = ratings["user_id"].value_counts().head(n_users).index
    sample_movies = ratings["movie_id"].value_counts().head(n_movies).index

    sub = ratings[
        ratings["user_id"].isin(sample_users) & ratings["movie_id"].isin(sample_movies)
    ]
    pivot = sub.pivot_table(index="user_id", columns="movie_id", values="rating")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = pivot.isna()
    sns.heatmap(
        pivot.fillna(0),
        mask=mask,
        cmap="mako",
        ax=ax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Rating"},
    )
    density = (~mask).values.mean()
    ax.set_title(
        f"User-Item Matrix (top {n_users} users × top {n_movies} movies)\n"
        f"Density: {density:.2%}",
        fontweight="bold",
    )
    ax.set_xlabel("Movie ID")
    ax.set_ylabel("User ID")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUTS / "sparsity_heatmap.png", dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Run all EDA
# ---------------------------------------------------------------------------

def run_all(
    ratings: pd.DataFrame,
    movies: Optional[pd.DataFrame] = None,
) -> None:
    print_summary(ratings, movies)
    plot_rating_distribution(ratings)
    plot_user_activity(ratings)
    plot_content_popularity(ratings, movies)
    plot_temporal_trends(ratings)
    plot_sparsity_heatmap(ratings)
    print(f"\nAll EDA charts saved to {OUTPUTS}")
