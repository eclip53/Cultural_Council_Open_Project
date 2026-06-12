#!/usr/bin/env python3
"""
scripts/generate_recs.py
=========================
Generate Top-K recommendations for a user and show case analysis.

Usage
-----
    python scripts/generate_recs.py --model svd --user_id 12345
    python scripts/generate_recs.py --model ncf --top_k 20
    python scripts/generate_recs.py --analyse   # success/failure analysis
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import get_dataset, train_test_split_temporal
from recommender import Recommender
from utils import load_model


def main():
    parser = argparse.ArgumentParser(description="Generate Netflix recommendations")
    parser.add_argument("--model", type=str, default="svd",
                        choices=["user_cf", "item_cf", "svd", "als", "ncf"])
    parser.add_argument("--user_id", type=int, default=None,
                        help="Specific user ID; if omitted, picks a random active user")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=500_000)
    parser.add_argument("--analyse", action="store_true",
                        help="Run success/failure case analysis")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    ratings, movies = get_dataset(sample_size=args.sample_size)
    train, test = train_test_split_temporal(ratings, test_frac=0.2)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    try:
        model = load_model(args.model)
        print(f"Loaded saved model: {args.model}")
    except FileNotFoundError:
        print(f"No saved model '{args.model}' found. Training a quick SVD …")
        from models import SVDModel
        model = SVDModel(n_factors=50, n_epochs=10)
        model.fit(train)

    rec = Recommender(model, movies=movies, ratings=ratings)

    # ------------------------------------------------------------------
    # Pick a user
    # ------------------------------------------------------------------
    if args.user_id is None:
        import numpy as np
        rng = np.random.default_rng(42)
        # Pick an active user from the test set
        test_users = test["user_id"].unique()
        args.user_id = int(rng.choice(test_users))

    # ------------------------------------------------------------------
    # Show recommendations
    # ------------------------------------------------------------------
    rec.print_recommendations(args.user_id, top_k=args.top_k)

    # Show user history
    history = rec.user_history(args.user_id, n=5)
    if not history.empty:
        print(f"User {args.user_id}'s recent ratings:")
        cols = [c for c in ["title", "movie_id", "rating", "date"] if c in history.columns]
        print(history[cols].to_string(index=False))
        print()

    # Explain first recommendation
    recs_df = rec.recommend(args.user_id, top_k=1)
    if not recs_df.empty:
        movie_id = int(recs_df.iloc[0]["movie_id"])
        explanation = rec.explain(args.user_id, movie_id)
        print(f"\nWhy '{recs_df.iloc[0]['title']}'?")
        print(f"  {explanation}\n")

    # ------------------------------------------------------------------
    # Success / failure analysis
    # ------------------------------------------------------------------
    if args.analyse:
        print("\n--- Success & Failure Case Analysis ---")
        cases = rec.analyse_recommendations(test, sample_users=50, top_k=args.top_k)

        print("\nSUCCESS CASES (recommendations that hit relevant movies):")
        for uid, recs, hits in cases["success"]:
            print(f"  User {uid}  ({hits} hits in Top-{args.top_k})")

        print("\nFAILURE CASES (no relevant movies in Top-K):")
        for uid, recs, hits in cases["failure"]:
            print(f"  User {uid}  ({hits} hits)")


if __name__ == "__main__":
    main()
