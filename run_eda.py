#!/usr/bin/env python3
"""
scripts/run_eda.py
==================
Standalone EDA pipeline.

Usage
-----
    python scripts/run_eda.py
    python scripts/run_eda.py --sample_size 200000
    python scripts/run_eda.py --raw_dir /path/to/data/raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import get_dataset
from eda import run_all


def main():
    parser = argparse.ArgumentParser(description="Netflix Prize EDA")
    parser.add_argument("--sample_size", type=int, default=500_000)
    parser.add_argument("--raw_dir", type=str, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.raw_dir:
        kwargs["raw_dir"] = args.raw_dir

    ratings, movies = get_dataset(sample_size=args.sample_size, **kwargs)
    run_all(ratings, movies)


if __name__ == "__main__":
    main()
