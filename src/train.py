import json
import pickle
import numpy as np
import polars as pl
import lightgbm as lgb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import LGB_PARAMS, get_run_paths, new_run_dir

FEATURES = [
    "log_eta", "hour_sin", "hour_cos", "dow", "is_weekend",
    "is_special_day", "demand_town_hour",
    "start_h3_te", "start_town_te", "did_hash_te", "uid_hash_te",
]


def to_xy(df: pl.DataFrame, target_col: str):
    avail = [f for f in FEATURES if f in df.columns]
    X = df.select(avail).to_pandas()
    y = df[target_col].to_numpy()
    return X, y, avail


def compute_metrics(actual: np.ndarray, pred_seconds: np.ndarray) -> dict:
    err = pred_seconds - actual
    ae = np.abs(err)
    return {
        "mae":              float(ae.mean()),
        "rmse":             float(np.sqrt((err**2).mean())),
        "mean_error":       float(err.mean()),
        "p50_ae":           float(np.percentile(ae, 50)),
        "p80_ae":           float(np.percentile(ae, 80)),
        "p90_ae":           float(np.percentile(ae, 90)),
        "p95_ae":           float(np.percentile(ae, 95)),
        "within_60":        float((ae <= 60).mean() * 100),
        "within_120":       float((ae <= 120).mean() * 100),
        "overpromise_gt60": float((err > 60).mean() * 100),
        "underpromise_gt60":float((err < -60).mean() * 100),
    }


def print_metrics(name: str, m: dict):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  MAE:    {m['mae']:.1f}s    RMSE: {m['rmse']:.1f}s")
    print(f"  MeanErr:{m['mean_error']:+.1f}s")
    print(f"  P50ae:  {m['p50_ae']:.0f}s  P80: {m['p80_ae']:.0f}s  P90: {m['p90_ae']:.0f}s  P95: {m['p95_ae']:.0f}s")
    print(f"  Within60s: {m['within_60']:.1f}%  Within120s: {m['within_120']:.1f}%")
    print(f"  Late>60s:  {m['overpromise_gt60']:.1f}%  Early>60s: {m['underpromise_gt60']:.1f}%")


def main(run_dir: Path = None, target_col: str = "target_logratio"):
    if run_dir is None:
        run_dir = new_run_dir()
    paths = get_run_paths(run_dir)
    MODELS_DIR  = paths["models_dir"]
    METRICS_DIR = paths["metrics_dir"]

    print("=== train.py ===")
    df = pl.read_parquet(paths["data_feats"])

    tr = df.filter(pl.col("split") == "train")
    va = df.filter(pl.col("split") == "valid")
    te = df.filter(pl.col("split") == "test")
    print(f"Train: {tr.height:,}  Valid: {va.height:,}  Test: {te.height:,}")

    Xtr, ytr, avail_feats = to_xy(tr, target_col)
    Xva, yva, _           = to_xy(va, target_col)
    Xte, yte, _           = to_xy(te, target_col)
    print(f"Features used: {avail_feats}")

    actual_tr = tr["time_accept_to_arrive"].to_numpy()
    actual_va = va["time_accept_to_arrive"].to_numpy()
    actual_te = te["time_accept_to_arrive"].to_numpy()
    eta_tr    = tr["driver_eta"].to_numpy()
    eta_va    = va["driver_eta"].to_numpy()
    eta_te    = te["driver_eta"].to_numpy()

    all_metrics = {}

    # ── Baseline ──────────────────────────────────────────────
    baseline_m = compute_metrics(actual_te, eta_te.astype(float))
    all_metrics["Baseline (driver_eta)"] = baseline_m
    print_metrics("Baseline (driver_eta)", baseline_m)

    # ── LightGBM 點估計 ───────────────────────────────────────
    print("\nTraining LightGBM...")
    model = lgb.LGBMRegressor(objective="l2", **LGB_PARAMS)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # 還原: log-ratio → corrected_eta = driver_eta * exp(pred)
    def restore(eta: np.ndarray, pred: np.ndarray) -> np.ndarray:
        if target_col == "target_logratio":
            return eta * np.exp(pred)
        else:
            return np.maximum(0, eta + pred)

    pred_te = restore(eta_te, model.predict(Xte))
    lgb_m = compute_metrics(actual_te, pred_te)
    all_metrics["LightGBM"] = lgb_m
    print_metrics("LightGBM", lgb_m)

    # Feature importance
    fi = sorted(zip(avail_feats, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for fname, imp in fi:
        print(f"  {fname:30s} {imp:6.0f}")

    # 存模型
    with open(MODELS_DIR / "lgb_main.pkl", "wb") as f:
        pickle.dump(model, f)

    # ── 分位數模型 ─────────────────────────────────────────────
    qmodels = {}
    for q in [0.1, 0.5, 0.9]:
        print(f"\nTraining quantile q={q}...")
        qm = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS)
        qm.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        qmodels[q] = qm

    q_preds = {q: restore(eta_te, qmodels[q].predict(Xte)) for q in [0.1, 0.5, 0.9]}

    # 修正交叉:確保 q10 <= q50 <= q90
    lo  = q_preds[0.1]
    mid = q_preds[0.5]
    hi  = q_preds[0.9]
    lo, mid, hi = np.sort(np.stack([lo, mid, hi], axis=1), axis=1).T

    # Coverage: 實際落在 [q10, q90] 的比例
    coverage = ((actual_te >= lo) & (actual_te <= hi)).mean() * 100
    print(f"\nQuantile coverage [q10, q90]: {coverage:.1f}%  (target ~80%)")

    # q50 當點估計時的指標
    q50_m = compute_metrics(actual_te, mid)
    all_metrics["Quantile q50"] = q50_m
    print_metrics("Quantile q50", q50_m)

    # q90 (保守估計,「寧早勿晚」)
    q90_m = compute_metrics(actual_te, hi)
    all_metrics["Quantile q90 (conservative)"] = q90_m
    print_metrics("Quantile q90 (conservative)", q90_m)

    with open(MODELS_DIR / "qmodels.pkl", "wb") as f:
        pickle.dump(qmodels, f)

    # ── 存所有指標 ─────────────────────────────────────────────
    with open(METRICS_DIR / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {METRICS_DIR / 'all_metrics.json'}")

    # ── 改善幅度摘要 ───────────────────────────────────────────
    bm = all_metrics["Baseline (driver_eta)"]
    print("\n" + "="*60)
    print("  SUMMARY vs Baseline")
    print("="*60)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'P90':>8} {'MAE Imp':>8} {'RMSE Imp':>9}")
    for name, m in all_metrics.items():
        mae_imp  = (bm["mae"]  - m["mae"])  / bm["mae"]  * 100
        rmse_imp = (bm["rmse"] - m["rmse"]) / bm["rmse"] * 100
        print(f"{name:<30} {m['mae']:>8.1f} {m['rmse']:>8.1f} {m['p90_ae']:>8.0f} {mae_imp:>7.1f}% {rmse_imp:>8.1f}%")

    # 回傳供 evaluate.py 用
    return model, qmodels, all_metrics, avail_feats


if __name__ == "__main__":
    main()
