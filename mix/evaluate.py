"""Evaluation — permutation importance + all comparison figures."""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_FEATS, MODELS_DIR, METRICS_DIR, FIGURES_DIR,
    FEATURES_MIX, CAT_FEATURES, SEED,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    ae  = np.abs(err)
    return {
        "mae":               float(ae.mean()),
        "rmse":              float(np.sqrt((err ** 2).mean())),
        "mean_error":        float(err.mean()),
        "p50_ae":            float(np.percentile(ae, 50)),
        "p80_ae":            float(np.percentile(ae, 80)),
        "p90_ae":            float(np.percentile(ae, 90)),
        "p95_ae":            float(np.percentile(ae, 95)),
        "within_60":         float((ae <= 60).mean() * 100),
        "within_120":        float((ae <= 120).mean() * 100),
        "overpromise_gt60":  float((err > 60).mean() * 100),
        "underpromise_gt60": float((err < -60).mean() * 100),
    }


def restore(eta: np.ndarray, pred_lr: np.ndarray) -> np.ndarray:
    return eta * np.exp(pred_lr)


# ── Permutation Importance ────────────────────────────────────────────────────

def permutation_importance(model, X_val, y_val, eta_val, actual_val,
                            n_repeats: int = 5) -> dict:
    """
    Permutation importance measured on MAE of corrected_eta (in seconds).
    Each feature is shuffled n_repeats times; we report mean MAE increase.
    This is the unbiased importance — not distorted by target formula or cardinality.
    """
    avail = list(X_val.columns)
    pred_base = restore(eta_val, model.predict(X_val))
    base_mae  = float(np.abs(pred_base - actual_val).mean())

    results = {}
    rng = np.random.default_rng(SEED)
    for feat in avail:
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_vals = rng.permutation(X_perm[feat].values)
            # Preserve Categorical dtype if present (required by LightGBM)
            if hasattr(X_perm[feat], "cat"):
                X_perm[feat] = pd.Categorical(
                    perm_vals, categories=X_perm[feat].cat.categories
                )
            else:
                X_perm[feat] = perm_vals
            pred_perm = restore(eta_val, model.predict(X_perm))
            perm_mae  = float(np.abs(pred_perm - actual_val).mean())
            deltas.append(perm_mae - base_mae)
        results[feat] = {
            "mean_mae_increase": float(np.mean(deltas)),
            "std_mae_increase":  float(np.std(deltas)),
        }

    print(f"\nPermutation importance (base MAE={base_mae:.2f}s, {n_repeats} repeats):")
    sorted_res = sorted(results.items(), key=lambda x: -x[1]["mean_mae_increase"])
    total = sum(max(v["mean_mae_increase"], 0) for _, v in sorted_res)
    for feat, v in sorted_res:
        pct = v["mean_mae_increase"] / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 1.5)
        print(f"  {feat:30s}  +{v['mean_mae_increase']:5.2f}s ({pct:5.1f}%)  {bar}")

    return {"base_mae": base_mae, "features": results}


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_calibration(actual, pred, name, ax):
    bins = np.percentile(pred, np.linspace(5, 95, 20))
    bins = np.unique(bins)
    idx  = np.digitize(pred, bins)
    xs, ys, ns = [], [], []
    for i in range(1, len(bins) + 1):
        mask = idx == i
        if mask.sum() >= 10:
            xs.append(pred[mask].mean()); ys.append(actual[mask].mean()); ns.append(mask.sum())
    xs, ys = np.array(xs), np.array(ys)
    ax.scatter(xs, ys, s=np.array(ns)/ns[0]*50+10, alpha=0.7, label=name)
    lim = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Predicted ETA (s)"); ax.set_ylabel("Mean Actual (s)")
    ax.set_title(f"Calibration: {name}"); ax.legend(fontsize=8)


def save_figures(actual, eta, pred_lgb, pred_q50, lo, hi,
                 hours, perm_result, fi_gain, avail, figures_dir: Path):

    # 1. Calibration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_calibration(actual, eta.astype(float), "Baseline", axes[0])
    plot_calibration(actual, pred_lgb,           "mix LightGBM", axes[1])
    plt.tight_layout(); plt.savefig(figures_dir / "calibration.png", dpi=150); plt.close()

    # 2. Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, err, label, color in [
        (axes[0], eta.astype(float) - actual, "Baseline",       "steelblue"),
        (axes[1], pred_lgb - actual,           "mix LightGBM",  "darkorange"),
    ]:
        ax.hist(np.clip(err, -600, 600), bins=80, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0, color="red", lw=1.5, ls="--")
        ax.axvline(np.median(err), color="green", lw=1.2, ls=":",
                   label=f"Median={np.median(err):.0f}s")
        ax.set_title(f"Error: {label}"); ax.set_xlabel("Predicted − Actual (s)")
        ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(figures_dir / "error_distribution.png", dpi=150); plt.close()

    # 3. MAE by hour
    if hours is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, pred, label in [(axes[0], eta.astype(float), "Baseline"),
                                 (axes[1], pred_lgb, "mix LightGBM")]:
            hour_mae = {h: np.abs(pred[hours==h] - actual[hours==h]).mean()
                        for h in range(24) if (hours==h).sum() > 0}
            ax.bar(list(hour_mae.keys()), list(hour_mae.values()), color="teal", alpha=0.7)
            ax.set_xlabel("Hour (Taiwan)"); ax.set_ylabel("MAE (s)"); ax.set_title(f"MAE by Hour: {label}")
        plt.tight_layout(); plt.savefig(figures_dir / "mae_by_hour.png", dpi=150); plt.close()

    # 4. Permutation importance (main)
    pf = perm_result["features"]
    sorted_feats = sorted(pf.items(), key=lambda x: -x[1]["mean_mae_increase"])
    names = [f for f, _ in sorted_feats]
    vals  = [v["mean_mae_increase"] for _, v in sorted_feats]
    errs  = [v["std_mae_increase"]  for _, v in sorted_feats]
    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*0.45)))
    colors  = ["#e07b54" if v >= 0 else "#aaaaaa" for v in vals]
    ax.barh(names[::-1], vals[::-1], xerr=errs[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean MAE increase when shuffled (s)")
    ax.set_title("Permutation Importance (mix, validation set)")
    plt.tight_layout(); plt.savefig(figures_dir / "permutation_importance.png", dpi=150); plt.close()

    # 5. Gain importance (for reference)
    total_gain = sum(fi_gain.values())
    gi_sorted  = sorted(fi_gain.items(), key=lambda x: x[1])
    gnames = [f for f, _ in gi_sorted]
    gvals  = [v / total_gain * 100 for _, v in gi_sorted]
    fig, ax = plt.subplots(figsize=(9, max(4, len(gnames)*0.45)))
    ax.barh(gnames, gvals, color="cornflowerblue", alpha=0.85)
    ax.set_xlabel("Gain importance (%)"); ax.set_title("Gain Importance (mix)")
    plt.tight_layout(); plt.savefig(figures_dir / "gain_importance.png", dpi=150); plt.close()

    # 6. Quantile coverage
    coverage = ((actual >= lo) & (actual <= hi)).mean() * 100
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(actual), min(300, len(actual)), replace=False)
    idx = idx[np.argsort(pred_q50[idx])]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(len(idx)), lo[idx], hi[idx], alpha=0.3, label="q10–q90 interval")
    ax.plot(range(len(idx)), pred_q50[idx], lw=1, color="orange", label="q50 (median)")
    ax.scatter(range(len(idx)), actual[idx], s=4, color="red", alpha=0.4, label="Actual")
    ax.set_title(f"Quantile Coverage: {coverage:.1f}%")
    ax.set_xlabel("Sample (sorted by q50)"); ax.set_ylabel("ETA (s)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(figures_dir / "quantile_coverage.png", dpi=150); plt.close()

    print(f"Figures saved to {figures_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== mix/evaluate.py ===")
    df = pl.read_parquet(DATA_FEATS)
    te = df.filter(pl.col("split") == "test")
    va = df.filter(pl.col("split") == "valid")
    print(f"Test rows: {te.height:,}   Valid rows: {va.height:,}")

    actual_te  = te["time_accept_to_arrive"].to_numpy()
    eta_te     = te["driver_eta"].to_numpy()
    actual_va  = va["time_accept_to_arrive"].to_numpy()
    eta_va     = va["driver_eta"].to_numpy()

    avail  = [f for f in FEATURES_MIX if f in te.columns]
    Xte    = te.select(avail).to_pandas()
    Xva    = va.select(avail).to_pandas()

    if "end_town" in avail:
        all_cats = sorted(set(Xte["end_town"].tolist()) | set(Xva["end_town"].tolist()) | {"unknown"})
        for X in [Xte, Xva]:
            X["end_town"] = pd.Categorical(X["end_town"], categories=all_cats)

    with open(MODELS_DIR / "lgb_main.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "qmodels.pkl", "rb") as f:
        qmodels = pickle.load(f)

    pred_lgb = restore(eta_te, model.predict(Xte))
    q_preds  = {q: restore(eta_te, qmodels[q].predict(Xte)) for q in [0.1, 0.5, 0.9]}
    lo, mid, hi = np.sort(
        np.stack([q_preds[0.1], q_preds[0.5], q_preds[0.9]], axis=1), axis=1
    ).T

    hours = te["hour"].to_numpy() if "hour" in te.columns else None

    # ── Permutation importance on validation set ──────────────────────────────
    print("\nRunning permutation importance (validation set, 5 repeats)...")
    perm_result = permutation_importance(
        model, Xva, None, eta_va, actual_va, n_repeats=5
    )
    with open(METRICS_DIR / "permutation_importance.json", "w") as f:
        json.dump(perm_result, f, indent=2)

    # Gain importance (for comparison)
    fi_gain  = dict(zip(model.feature_name(), model.feature_importance("gain").tolist()))
    fi_split = dict(zip(model.feature_name(), model.feature_importance("split").tolist()))
    total_g  = sum(fi_gain.values())
    print(f"\nGain importance (top 10):")
    for feat, g in sorted(fi_gain.items(), key=lambda x: -x[1])[:10]:
        pct = g / total_g * 100
        print(f"  {feat:30s}  {pct:5.1f}%")

    # ── Figures ───────────────────────────────────────────────────────────────
    save_figures(
        actual_te, eta_te, pred_lgb, mid, lo, hi,
        hours, perm_result, fi_gain, avail, FIGURES_DIR
    )

    # ── Print full metrics table ──────────────────────────────────────────────
    with open(METRICS_DIR / "all_metrics.json") as f:
        all_m = json.load(f)

    print(f"\n{'='*90}")
    print(f"{'Model':<32} {'MAE':>7} {'RMSE':>7} {'MeanErr':>9} {'W60%':>7} {'W120%':>6} "
          f"{'P90':>6} {'Late>60':>8}")
    print("-"*90)
    for name, m in all_m.items():
        print(f"{name:<32} {m['mae']:>7.1f} {m['rmse']:>7.1f} {m['mean_error']:>+9.1f} "
              f"{m['within_60']:>7.1f} {m['within_120']:>6.1f} {m['p90_ae']:>6.0f} "
              f"{m['overpromise_gt60']:>7.1f}%")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
