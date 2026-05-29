import json
import pickle
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_FEATS, OUT

FIGURES_DIR = OUT / "figures"
METRICS_DIR = OUT / "metrics"
MODELS_DIR  = OUT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "log_eta", "hour_sin", "hour_cos", "dow", "is_weekend",
    "is_special_day", "demand_town_hour",
    "start_h3_te", "start_town_te", "did_hash_te", "uid_hash_te",
]


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    ae = np.abs(err)
    return {
        "mae":              ae.mean(),
        "rmse":             np.sqrt((err**2).mean()),
        "mean_error":       err.mean(),
        "p50_ae":           np.percentile(ae, 50),
        "p80_ae":           np.percentile(ae, 80),
        "p90_ae":           np.percentile(ae, 90),
        "p95_ae":           np.percentile(ae, 95),
        "within_60":        (ae <= 60).mean() * 100,
        "within_120":       (ae <= 120).mean() * 100,
        "overpromise_gt60": (err > 60).mean() * 100,
        "underpromise_gt60":(err < -60).mean() * 100,
    }


def plot_calibration(actual, pred, name, ax):
    bins = np.percentile(pred, np.linspace(5, 95, 20))
    bins = np.unique(bins)
    idx = np.digitize(pred, bins)
    xs, ys, ns = [], [], []
    for i in range(1, len(bins) + 1):
        mask = idx == i
        if mask.sum() >= 10:
            xs.append(pred[mask].mean())
            ys.append(actual[mask].mean())
            ns.append(mask.sum())
    xs, ys = np.array(xs), np.array(ys)
    sc = ax.scatter(xs, ys, s=np.array(ns)/ns[0]*50+10, alpha=0.7, label=name)
    lim = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set_xlabel("Predicted ETA (s)")
    ax.set_ylabel("Mean Actual Time (s)")
    ax.set_title(f"Calibration: {name}")
    ax.legend(fontsize=8)


def slice_metrics(df: pl.DataFrame, actual: np.ndarray, pred: np.ndarray,
                  group_col: str, top_n: int = 15) -> pl.DataFrame:
    df2 = df.with_columns([
        pl.Series("_actual", actual),
        pl.Series("_pred",   pred),
    ])
    rows = []
    for key, grp in df2.group_by(group_col):
        a = grp["_actual"].to_numpy()
        p = grp["_pred"].to_numpy()
        m = metrics(a, p)
        rows.append({group_col: key[0] if isinstance(key, tuple) else key,
                     "n": len(a), **m})
    result = pl.DataFrame(rows).sort("n", descending=True).head(top_n)
    return result


def main(target_col: str = "target_logratio"):
    print("=== evaluate.py ===")
    df = pl.read_parquet(DATA_FEATS)
    te = df.filter(pl.col("split") == "test")
    print(f"Test rows: {te.height:,}")

    actual  = te["time_accept_to_arrive"].to_numpy()
    eta     = te["driver_eta"].to_numpy()
    avail   = [f for f in FEATURES if f in te.columns]
    Xte     = te.select(avail).to_pandas()

    with open(MODELS_DIR / "lgb_main.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "qmodels.pkl", "rb") as f:
        qmodels = pickle.load(f)

    def restore(eta_arr, pred_arr):
        if target_col == "target_logratio":
            return eta_arr * np.exp(pred_arr)
        return np.maximum(0, eta_arr + pred_arr)

    pred_lgb = restore(eta, model.predict(Xte))
    q_preds = {q: restore(eta, qmodels[q].predict(Xte)) for q in [0.1, 0.5, 0.9]}
    lo, mid, hi = np.sort(np.stack([q_preds[0.1], q_preds[0.5], q_preds[0.9]], axis=1), axis=1).T

    # ── 圖1: Calibration ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_calibration(actual, eta.astype(float), "Baseline", axes[0])
    plot_calibration(actual, pred_lgb,           "LightGBM", axes[1])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration.png", dpi=150)
    plt.close()
    print("Saved: calibration.png")

    # ── 圖2: Error distribution ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    err_base = eta.astype(float) - actual
    err_lgb  = pred_lgb - actual

    for ax, err, label, color in [
        (axes[0], err_base, "Baseline",  "steelblue"),
        (axes[1], err_lgb,  "LightGBM",  "darkorange"),
    ]:
        ax.hist(np.clip(err, -600, 600), bins=80, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0, color="red", lw=1.5, ls="--")
        ax.axvline(np.median(err), color="green", lw=1.2, ls=":", label=f"Median={np.median(err):.0f}s")
        ax.set_title(f"Error Distribution: {label}")
        ax.set_xlabel("Predicted - Actual (s)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}s"))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_distribution.png", dpi=150)
    plt.close()
    print("Saved: error_distribution.png")

    # ── 圖3: 時段切片 ───────────────────────────────────────────────────
    if "hour" in te.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        hours = te["hour"].to_numpy()
        for ax, pred, label in [(axes[0], eta.astype(float), "Baseline"),
                                 (axes[1], pred_lgb, "LightGBM")]:
            hour_mae = {}
            for h in range(24):
                mask = hours == h
                if mask.sum() > 0:
                    hour_mae[h] = np.abs(pred[mask] - actual[mask]).mean()
            ax.bar(list(hour_mae.keys()), list(hour_mae.values()), color="teal", alpha=0.7)
            ax.set_xlabel("Hour of Day (Taiwan)")
            ax.set_ylabel("MAE (s)")
            ax.set_title(f"MAE by Hour: {label}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "mae_by_hour.png", dpi=150)
        plt.close()
        print("Saved: mae_by_hour.png")

    # ── 圖4: Feature importance ─────────────────────────────────────────
    fi = sorted(zip(avail, model.feature_importances_), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f for f, _ in fi], [v for _, v in fi], color="cornflowerblue")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("LightGBM Feature Importances")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

    # ── 圖5: Quantile coverage ──────────────────────────────────────────
    coverage = ((actual >= lo) & (actual <= hi)).mean() * 100
    interval_width = (hi - lo).mean()
    print(f"\nQuantile [q10, q90] coverage: {coverage:.1f}%  avg width: {interval_width:.0f}s")

    fig, ax = plt.subplots(figsize=(10, 4))
    sample_idx = np.random.choice(len(actual), min(300, len(actual)), replace=False)
    sample_idx = sample_idx[np.argsort(mid[sample_idx])]
    x = np.arange(len(sample_idx))
    ax.fill_between(x, lo[sample_idx], hi[sample_idx], alpha=0.3, label="q10–q90 interval")
    ax.plot(x, mid[sample_idx], lw=1, color="orange", label="q50 (median)")
    ax.scatter(x, actual[sample_idx], s=4, color="red", alpha=0.4, label="Actual")
    ax.set_xlabel("Sample (sorted by q50)")
    ax.set_ylabel("ETA (s)")
    ax.set_title(f"Quantile Interval Coverage: {coverage:.1f}%")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quantile_coverage.png", dpi=150)
    plt.close()
    print("Saved: quantile_coverage.png")

    # ── 切片分析:top towns ──────────────────────────────────────────────
    if "start_town" in te.columns:
        slice_df = slice_metrics(te, actual, pred_lgb, "start_town", top_n=15)
        slice_df.write_csv(METRICS_DIR / "slice_by_town.csv")
        print(f"Saved: slice_by_town.csv")
        print(slice_df.select(["start_town","n","mae","rmse","p90_ae","overpromise_gt60"]))

    # ── 特殊日切片 ──────────────────────────────────────────────────────
    if "is_special_day" in te.columns:
        slice_sp = slice_metrics(te, actual, pred_lgb, "is_special_day", top_n=5)
        print(f"\nSpecial day slice:\n{slice_sp.select(['is_special_day','n','mae','rmse','p90_ae'])}")

    # ── 主結果表 ─────────────────────────────────────────────────────────
    with open(METRICS_DIR / "all_metrics.json") as f:
        all_m = json.load(f)

    print("\n" + "="*85)
    print(f"{'Model':<32} {'MAE':>7} {'RMSE':>7} {'MeanErr':>8} {'W60%':>6} {'W120%':>6} {'P90':>6} {'Late>60':>8}")
    print("-"*85)
    for name, m in all_m.items():
        print(f"{name:<32} {m['mae']:>7.1f} {m['rmse']:>7.1f} {m['mean_error']:>+8.1f} "
              f"{m['within_60']:>6.1f} {m['within_120']:>6.1f} {m['p90_ae']:>6.0f} "
              f"{m['overpromise_gt60']:>7.1f}%")
    print("="*85)


if __name__ == "__main__":
    main()
