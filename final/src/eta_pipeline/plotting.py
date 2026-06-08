"""Optional PNG plots (flagged off by default)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plots: pip install matplotlib")


def residual_histogram(
    actual: np.ndarray,
    corrected: np.ndarray,
    raw: np.ndarray,
    out_path: Path,
) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-600, 600, 80)
    ax.hist(raw - actual, bins=bins, alpha=0.5, label="Raw API error", color="steelblue")
    ax.hist(corrected - actual, bins=bins, alpha=0.5, label="Corrected error", color="salmon")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Prediction error (corrected − actual) [s]")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution: Raw vs Corrected ETA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def importance_bar(importance_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    plt = _mpl()
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_xlabel("Gain importance")
    ax.set_title(f"Feature importance (top {top_n})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def error_vs_eta(
    driver_eta: np.ndarray,
    error: np.ndarray,
    out_path: Path,
    sample: int = 5000,
) -> None:
    plt = _mpl()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(driver_eta), size=min(sample, len(driver_eta)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(driver_eta[idx] / 60, error[idx], alpha=0.15, s=5, color="steelblue")
    ax.axhline(0, color="red", linewidth=0.8)
    ax.set_xlabel("driver_eta (minutes)")
    ax.set_ylabel("Corrected error [s]")
    ax.set_title("Corrected error vs driver_eta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def save_all(
    actual: np.ndarray,
    corrected: np.ndarray,
    raw: np.ndarray,
    importance_df: Optional[pd.DataFrame],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_histogram(actual, corrected, raw, out_dir / "residuals.png")
    error_vs_eta(raw, corrected - actual, out_dir / "error_vs_eta.png")
    if importance_df is not None:
        importance_bar(importance_df, out_dir / "feature_importance.png")
