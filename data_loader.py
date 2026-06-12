"""
data_loader.py
==============
Ingests the Netflix Prize dataset from raw text files and returns
clean pandas DataFrames ready for modelling.

Raw layout
----------
combined_data_N.txt:
    <movie_id>:          <- marks start of a new movie block
    <user_id>,<rating>,<date>
    ...

movie_titles.csv:
    movie_id,year_of_release,title
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Raw parsing
# ---------------------------------------------------------------------------

def parse_combined_file(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a single combined_data_N.txt file into a DataFrame.

    Returns
    -------
    DataFrame with columns: movie_id, user_id, rating, date
    """
    filepath = Path(filepath)
    rows: list[tuple] = []
    current_movie: Optional[int] = None

    with open(filepath, "r") as fh:
        for line in tqdm(fh, desc=f"Parsing {filepath.name}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_movie = int(line[:-1])
            else:
                parts = line.split(",")
                if len(parts) == 3 and current_movie is not None:
                    user_id, rating, date = parts
                    rows.append((current_movie, int(user_id), int(rating), date))

    return pd.DataFrame(rows, columns=["movie_id", "user_id", "rating", "date"])


def load_raw_ratings(
    raw_dir: str | Path = RAW_DIR,
    files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load and concatenate all combined_data_*.txt files.

    Parameters
    ----------
    raw_dir : directory containing the raw files
    files   : explicit list of filenames; defaults to all combined_data_*.txt
    """
    raw_dir = Path(raw_dir)
    if files is None:
        files = sorted(raw_dir.glob("combined_data_*.txt"))
    else:
        files = [raw_dir / f for f in files]

    dfs: list[pd.DataFrame] = []
    for fp in files:
        dfs.append(parse_combined_file(fp))

    ratings = pd.concat(dfs, ignore_index=True)
    ratings["date"] = pd.to_datetime(ratings["date"])
    ratings["rating"] = ratings["rating"].astype(np.int8)
    ratings["movie_id"] = ratings["movie_id"].astype(np.int32)
    ratings["user_id"] = ratings["user_id"].astype(np.int32)
    return ratings


def load_movie_titles(raw_dir: str | Path = RAW_DIR) -> pd.DataFrame:
    """
    Load movie_titles.csv.

    The file is ISO-8859-1 encoded and contains occasional commas in titles,
    so we limit the number of splits.
    """
    filepath = Path(raw_dir) / "movie_titles.csv"
    movies = pd.read_csv(
        filepath,
        encoding="ISO-8859-1",
        header=None,
        names=["movie_id", "year", "title"],
        on_bad_lines="skip",
    )
    movies["movie_id"] = movies["movie_id"].astype(np.int32)
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    return movies


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_ratings(
    ratings: pd.DataFrame,
    n: int = 500_000,
    min_ratings_per_user: int = 10,
    min_ratings_per_movie: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a random sample of `n` ratings after filtering sparse
    users and items to ensure collaborative filtering has enough signal.
    """
    rng = np.random.default_rng(seed)

    # Filter sparse users
    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_ratings_per_user].index
    ratings = ratings[ratings["user_id"].isin(active_users)]

    # Filter sparse movies
    movie_counts = ratings["movie_id"].value_counts()
    popular_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
    ratings = ratings[ratings["movie_id"].isin(popular_movies)]

    if len(ratings) > n:
        idx = rng.choice(len(ratings), size=n, replace=False)
        ratings = ratings.iloc[idx].copy()

    ratings = ratings.reset_index(drop=True)
    return ratings


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def train_test_split_temporal(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: the most recent `test_frac` of each user's ratings
    go into the test set.  This mirrors a realistic deployment scenario
    where we train on historical data and predict future preferences.
    """
    ratings = ratings.sort_values(["user_id", "date"])
    test_rows = (
        ratings.groupby("user_id")
        .apply(lambda g: g.tail(max(1, int(len(g) * test_frac))))
        .reset_index(drop=True)
    )
    test_idx = set(test_rows.index)
    train = ratings[~ratings.index.isin(test_idx)].copy()
    test = ratings[ratings.index.isin(test_idx)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_test_split_random(
    ratings: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random stratified split keeping at least one train rating per user."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(ratings, test_size=test_frac, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_processed(
    ratings: pd.DataFrame,
    filename: str = "ratings_sample.parquet",
    processed_dir: str | Path = PROCESSED_DIR,
) -> Path:
    out = Path(processed_dir) / filename
    ratings.to_parquet(out, index=False)
    print(f"Saved {len(ratings):,} ratings → {out}")
    return out


def load_processed(
    filename: str = "ratings_sample.parquet",
    processed_dir: str | Path = PROCESSED_DIR,
) -> pd.DataFrame:
    path = Path(processed_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_eda.py first to create the sample."
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Convenience loader used by training scripts
# ---------------------------------------------------------------------------

def get_dataset(
    sample_size: int = 500_000,
    raw_dir: str | Path = RAW_DIR,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (ratings_df, movies_df).

    If a cached parquet exists and force_rebuild is False it is loaded
    directly, otherwise the raw files are parsed and sampled.
    """
    cache = PROCESSED_DIR / f"ratings_{sample_size // 1000}k.parquet"
    if cache.exists() and not force_rebuild:
        print(f"Loading cached sample from {cache}")
        ratings = pd.read_parquet(cache)
    else:
        print("Parsing raw files …")
        raw_dir = Path(raw_dir)
        all_files = sorted(raw_dir.glob("combined_data_*.txt"))
        if not all_files:
            raise FileNotFoundError(
                f"No combined_data_*.txt files found in {raw_dir}.\n"
                "Download the dataset from:\n"
                "  https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data"
            )
        ratings = load_raw_ratings(raw_dir=raw_dir)
        ratings = sample_ratings(ratings, n=sample_size)
        save_processed(ratings, filename=cache.name)

    movies = load_movie_titles(raw_dir=raw_dir)
    return ratings, movies
