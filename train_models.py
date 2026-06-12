#!/usr/bin/env python3
"""
scripts/train_models.py
========================
Train all recommendation models, evaluate them, and produce comparison charts.

Usage
-----
    python scripts/train_models.py
    python scripts/train_models.py --sample_size 200000 --models svd als
    python scripts/train_models.py --skip_ncf   # skip NCF (no GPU)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import get_dataset, train_test_split_temporal
from evaluation import evaluate_model, compare_models
from utils import save_model, save_results, plot_metric_comparison, timer


def main():
    parser = argparse.ArgumentParser(description="Train Netflix recommendation models")
    parser.add_argument("--sample_size", type=int, default=500_000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_map_users", type=int, default=500,
                        help="Max users for MAP evaluation (speed)")
    parser.add_argument("--models", nargs="+",
                        default=["user_cf", "item_cf", "svd", "als", "ncf"],
                        help="Which models to train")
    parser.add_argument("--skip_ncf", action="store_true")
    args = parser.parse_args()

    if args.skip_ncf and "ncf" in args.models:
        args.models.remove("ncf")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data …")
    ratings, movies = get_dataset(sample_size=args.sample_size)
    train, test = train_test_split_temporal(ratings, test_frac=0.2)
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

    all_results: dict = {}

    # ------------------------------------------------------------------
    # User-Based CF
    # ------------------------------------------------------------------
    if "user_cf" in args.models:
        print("\n[2/5] User-Based Collaborative Filtering …")
        from models import UserCF

        with timer("UserCF fit"):
            ucf = UserCF(n_neighbours=50)
            ucf.fit(train)

        y_true = test["rating"].values
        with timer("UserCF predict"):
            y_pred = ucf.predict(test)

        results = evaluate_model(
            "User-Based CF",
            y_true, y_pred,
            test_df=test,
            recommend_fn=lambda uid, k: ucf.recommend(uid, k),
            k=args.top_k,
            max_map_users=args.max_map_users,
        )
        all_results["User-Based CF"] = results
        save_model(ucf, "user_cf")

    # ------------------------------------------------------------------
    # Item-Based CF
    # ------------------------------------------------------------------
    if "item_cf" in args.models:
        print("\n[3/5] Item-Based Collaborative Filtering …")
        from models import ItemCF

        with timer("ItemCF fit"):
            icf = ItemCF(n_neighbours=50)
            icf.fit(train)

        with timer("ItemCF predict"):
            y_pred = icf.predict(test)

        results = evaluate_model(
            "Item-Based CF",
            test["rating"].values, y_pred,
            test_df=test,
            recommend_fn=lambda uid, k: icf.recommend(uid, k),
            k=args.top_k,
            max_map_users=args.max_map_users,
        )
        all_results["Item-Based CF"] = results
        save_model(icf, "item_cf")

    # ------------------------------------------------------------------
    # SVD
    # ------------------------------------------------------------------
    if "svd" in args.models:
        print("\n[4/5] SVD Matrix Factorization …")
        try:
            from models import SVDModel

            with timer("SVD fit"):
                svd = SVDModel(n_factors=100, n_epochs=20)
                svd.fit(train)

            with timer("SVD predict"):
                y_pred = svd.predict(test)

            results = evaluate_model(
                "SVD",
                test["rating"].values, y_pred,
                test_df=test,
                recommend_fn=lambda uid, k: svd.recommend(uid, k),
                k=args.top_k,
                max_map_users=args.max_map_users,
            )
            all_results["SVD"] = results
            save_model(svd, "svd")
        except ImportError as e:
            print(f"  Skipping SVD: {e}")

    # ------------------------------------------------------------------
    # ALS
    # ------------------------------------------------------------------
    if "als" in args.models:
        print("\n[4/5] ALS Matrix Factorization …")
        from models import ALSModel

        with timer("ALS fit"):
            als = ALSModel(n_factors=50, n_iterations=15, reg=0.1)
            als.fit(train)

        with timer("ALS predict"):
            y_pred = als.predict(test)

        results = evaluate_model(
            "ALS",
            test["rating"].values, y_pred,
            test_df=test,
            recommend_fn=lambda uid, k: als.recommend(uid, k),
            k=args.top_k,
            max_map_users=args.max_map_users,
        )
        all_results["ALS"] = results
        save_model(als, "als")

    # ------------------------------------------------------------------
    # NCF
    # ------------------------------------------------------------------
    if "ncf" in args.models:
        print("\n[5/5] Neural Collaborative Filtering …")
        try:
            from models import NCFModel

            with timer("NCF fit"):
                ncf = NCFModel(
                    gmf_factors=32,
                    mlp_factors=32,
                    mlp_layers=[64, 32, 16],
                    lr=1e-3,
                    batch_size=1024,
                    n_epochs=10,
                )
                ncf.fit(train)

            with timer("NCF predict"):
                y_pred = ncf.predict(test)

            results = evaluate_model(
                "NCF",
                test["rating"].values, y_pred,
                test_df=test,
                recommend_fn=lambda uid, k: ncf.recommend(uid, k),
                k=args.top_k,
                max_map_users=args.max_map_users,
            )
            all_results["NCF"] = results
            save_model(ncf, "ncf")
        except (ImportError, Exception) as e:
            print(f"  Skipping NCF: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_results:
        df = compare_models(all_results)
        save_results(all_results)

        from pathlib import Path
        plot_metric_comparison(
            all_results, "RMSE",
            save_path=Path("outputs") / "rmse_comparison.png"
        )
        plot_metric_comparison(
            all_results, f"MAP@{args.top_k}",
            save_path=Path("outputs") / "map_comparison.png"
        )
        print(f"\nAll done. Results saved to outputs/")


if __name__ == "__main__":
    main()
