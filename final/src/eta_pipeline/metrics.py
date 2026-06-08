"""MAE/RMSE/bias/within-X/over-under-promise + segment analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def evaluate(
    actual: np.ndarray,
    corrected: np.ndarray,
    raw: Optional[np.ndarray] = None,
) -> dict:
    err = corrected - actual
    ae  = np.abs(err)
    result = {
        "mae":           float(ae.mean()),
        "rmse":          float(np.sqrt((err ** 2).mean())),
        "bias":          float(err.mean()),
        "within_60":     float((ae <= 60).mean() * 100),
        "within_120":    float((ae <= 120).mean() * 100),
        "p50":           float(np.percentile(ae, 50)),
        "p90":           float(np.percentile(ae, 90)),
        "p95":           float(np.percentile(ae, 95)),
        "over_promise":  float((err > 60).mean() * 100),
        "under_promise": float((err < -60).mean() * 100),
        "n":             int(len(actual)),
    }
    if raw is not None:
        raw_err = raw - actual
        raw_ae  = np.abs(raw_err)
        raw_mae  = float(raw_ae.mean())
        raw_rmse = float(np.sqrt((raw_err ** 2).mean()))
        result["mae_vs_raw"]  = result["mae"] - raw_mae
        result["rmse_vs_raw"] = result["rmse"] - raw_rmse
        if raw_mae > 0:
            result["mae_imp_pct"]  = (raw_mae  - result["mae"])  / raw_mae  * 100
            result["rmse_imp_pct"] = (raw_rmse - result["rmse"]) / raw_rmse * 100
    return result


def _bucket_col(df: pd.DataFrame, col: str, edges: List[float], label: str) -> pd.Series:
    """Bin a numeric column into labelled buckets."""
    labels = []
    for i, e in enumerate(edges):
        if i + 1 < len(edges):
            labels.append(f"{e}–{edges[i+1]}")
        else:
            labels.append(f"{e}+")
    return pd.cut(df[col], bins=edges + [np.inf], labels=labels, right=False)


def evaluate_by(
    df: pd.DataFrame,
    actual_col: str,
    corrected_col: str,
    raw_col: str,
    by: str,
    min_rows: int = 200,
) -> Dict[Any, dict]:
    results = {}
    for val, grp in df.groupby(by, observed=True):
        if len(grp) < min_rows:
            continue
        results[val] = evaluate(
            grp[actual_col].values,
            grp[corrected_col].values,
            grp[raw_col].values,
        )
    return results


def segment_analysis(
    df: pd.DataFrame,
    actual_col: str,
    corrected_col: str,
    raw_col: str,
    cfg,
) -> Dict[str, Dict[Any, dict]]:
    segs: Dict[str, Dict] = {}
    min_r = cfg.report.min_segment_rows

    for by in ["hour", "day_of_week", "start_county", "is_lunar_new_year"]:
        if by in df.columns:
            segs[by] = evaluate_by(df, actual_col, corrected_col, raw_col, by, min_r)

    # Distance buckets
    if "haversine_m" in df.columns:
        edges_km = cfg.report.distance_buckets_km
        edges_m  = [e * 1000 for e in edges_km]
        df = df.copy()
        df["_dist_bucket"] = _bucket_col(df, "haversine_m", edges_m, "dist")
        segs["distance_km"] = evaluate_by(df, actual_col, corrected_col, raw_col,
                                          "_dist_bucket", min_r)

    # driver_eta buckets
    edges_s = cfg.report.driver_eta_buckets_s
    df["_eta_bucket"] = _bucket_col(df, raw_col, edges_s, "eta")
    segs["driver_eta_bucket"] = evaluate_by(df, actual_col, corrected_col, raw_col,
                                            "_eta_bucket", min_r)

    return segs
