"""
utils.py
========
Shared utilities: logging, model persistence, plotting helpers.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = ""):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {label}: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, name: str, out_dir: str | Path = OUTPUTS / "models") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  Model saved → {path}")
    return path


def load_model(name: str, out_dir: str | Path = OUTPUTS / "models") -> Any:
    path = Path(out_dir) / f"{name}.pkl"
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Dict[str, float]], filename: str = "results.json") -> Path:
    path = OUTPUTS / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {path}")
    return path


def load_results(filename: str = "results.json") -> Dict[str, Dict[str, float]]:
    path = OUTPUTS / filename
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "RMSE",
    save_path: str | Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.mako(np.linspace(0.3, 0.85, len(models)))
    bars = ax.bar(models, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.4f}",
            ha="center",
            fontsize=10,
        )
    ax.set_title(f"Model Comparison — {metric}", fontweight="bold", fontsize=13)
    ax.set_ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
