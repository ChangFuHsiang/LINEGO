"""Training — Optuna 80 trials with asymmetric P90 objective + quantile models."""

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_FEATS, MODELS_DIR, METRICS_DIR,
    FEATURES_MIX, CAT_FEATURES,
    OPTUNA_N_TRIALS, OPTUNA_BOOST_ROUND, OPTUNA_EARLY_STOP,
    ROUND_INFLATION, OPTUNA_DB, SEED,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("WARNING: optuna not installed — will use default params")


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
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


def print_metrics(name: str, m: dict):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  MAE:    {m['mae']:.1f}s    RMSE: {m['rmse']:.1f}s")
    print(f"  MeanErr:{m['mean_error']:+.1f}s")
    print(f"  P50:{m['p50_ae']:.0f}s  P80:{m['p80_ae']:.0f}s  P90:{m['p90_ae']:.0f}s  P95:{m['p95_ae']:.0f}s")
    print(f"  Within60:{m['within_60']:.1f}%  Within120:{m['within_120']:.1f}%")
    print(f"  Late>60: {m['overpromise_gt60']:.1f}%   Early>60: {m['underpromise_gt60']:.1f}%")


def restore(eta: np.ndarray, pred_logratio: np.ndarray) -> np.ndarray:
    """log-ratio → corrected seconds."""
    return eta * np.exp(pred_logratio)


def to_xy(df: pl.DataFrame, feats: list[str]):
    avail = [f for f in feats if f in df.columns]
    X = df.select(avail).to_pandas()
    y = df["target_logratio"].to_numpy()
    return X, y, avail


# ── Optuna objective ──────────────────────────────────────────────────────────

def _make_objective(Xtr, ytr, Xva, yva, eta_va, actual_va, avail, eff_cat):
    """
    Minimize: P90(|err|) + 0.5 × late_penalty
    where err = corrected_eta - actual (positive=early, negative=late).
    Late arrivals (err < -60s) incur double penalty.
    """
    def objective(trial):
        params = {
            "objective":        "regression",
            "metric":           "rmse",
            "verbosity":        -1,
            "seed":             SEED,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples":trial.suggest_int("min_child_samples", 20, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1":        trial.suggest_float("lambda_l1", 1e-5, 10.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-5, 10.0, log=True),
        }
        dtrain = lgb.Dataset(Xtr, ytr,
                             categorical_feature=eff_cat or "auto",
                             free_raw_data=False)
        dval   = lgb.Dataset(Xva, yva,
                             categorical_feature=eff_cat or "auto",
                             reference=dtrain, free_raw_data=False)
        model = lgb.train(
            params, dtrain,
            num_boost_round=OPTUNA_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(OPTUNA_EARLY_STOP, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        trial.set_user_attr("best_iteration", int(model.best_iteration))

        pred_lr   = model.predict(Xva, num_iteration=model.best_iteration)
        corrected = restore(eta_va, pred_lr)
        err       = corrected - actual_va         # positive = early (ok), negative = late (bad)
        abs_err   = np.abs(err)
        p90       = float(np.percentile(abs_err, 90))
        # Double penalty for late arrivals > 60s
        late_pen  = float(np.where(err < -60, abs_err * 2, abs_err).mean())
        return p90 + 0.5 * late_pen

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== mix/train.py ===")
    df  = pl.read_parquet(DATA_FEATS)
    tr  = df.filter(pl.col("split") == "train")
    va  = df.filter(pl.col("split") == "valid")
    te  = df.filter(pl.col("split") == "test")
    print(f"Train:{tr.height:,}  Valid:{va.height:,}  Test:{te.height:,}")

    Xtr, ytr, avail = to_xy(tr, FEATURES_MIX)
    Xva, yva, _     = to_xy(va, FEATURES_MIX)
    Xte, yte, _     = to_xy(te, FEATURES_MIX)
    print(f"Features used ({len(avail)}): {avail}")

    # end_town must be pandas Categorical for LightGBM native categorical handling
    if "end_town" in avail:
        all_cats = sorted(set(Xtr["end_town"].tolist()) | {"unknown"})
        for X in [Xtr, Xva, Xte]:
            X["end_town"] = pd.Categorical(X["end_town"], categories=all_cats)

    actual_tr  = tr["time_accept_to_arrive"].to_numpy()
    actual_va  = va["time_accept_to_arrive"].to_numpy()
    actual_te  = te["time_accept_to_arrive"].to_numpy()
    eta_tr     = tr["driver_eta"].to_numpy()
    eta_va     = va["driver_eta"].to_numpy()
    eta_te     = te["driver_eta"].to_numpy()

    eff_cat = [c for c in CAT_FEATURES if c in avail]

    all_metrics = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_m = compute_metrics(actual_te, eta_te.astype(float))
    all_metrics["Baseline (driver_eta)"] = baseline_m
    print_metrics("Baseline (driver_eta)", baseline_m)

    # ── Fixed best params (from final/ pipeline 30-trial Optuna, SESSION_SUMMARY) ──
    # Optuna skipped for now to validate mix feature design first.
    # Source: artifacts/best_params_20260601-150143-6ef231d3.json (30 trials)
    best_iteration = 1839
    best_params = {
        "objective":        "regression",
        "metric":           "rmse",
        "verbosity":        -1,
        "seed":             SEED,
        "learning_rate":    0.0499,
        "num_leaves":       120,
        "min_child_samples":262,
        "feature_fraction": 0.605,
        "bagging_fraction": 0.945,
        "bagging_freq":     3,
        "lambda_l1":        3.46e-4,
        "lambda_l2":        0.0677,
    }
    print(f"\nUsing fixed best params (from final/ 30-trial Optuna):")
    for k, v in best_params.items():
        if k not in ("objective", "metric", "verbosity", "seed"):
            print(f"  {k}: {v}")

    # ── Final model: retrain on train+valid with best params ──────────────────
    X_tv    = pd.concat([Xtr, Xva], axis=0)
    y_tv    = np.concatenate([ytr, yva])
    eta_tv  = np.concatenate([eta_tr, eta_va])
    actual_tv = np.concatenate([actual_tr, actual_va])

    # Cap at 1000 rounds: best_iteration=1839 was found on different feature set;
    # using fixed params so we don't know the correct iteration count for this mix.
    final_rounds = min(int(best_iteration * ROUND_INFLATION), 1000)
    print(f"\nFinal retrain on train+valid: {final_rounds} rounds")

    if best_params:
        dtv = lgb.Dataset(X_tv, y_tv,
                          categorical_feature=eff_cat or "auto",
                          free_raw_data=False)
        final_model = lgb.train(
            best_params, dtv,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(100)],
        )
    else:
        from config import LGB_DEFAULT
        final_model = lgb.LGBMRegressor(
            objective="regression_l1", **LGB_DEFAULT
        )
        final_model.fit(
            X_tv, y_tv,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)],
        )
        final_model = final_model.booster_

    pred_te = restore(eta_te, final_model.predict(Xte))
    mix_m   = compute_metrics(actual_te, pred_te)
    all_metrics["mix LightGBM"] = mix_m
    print_metrics("mix LightGBM (test)", mix_m)

    with open(MODELS_DIR / "lgb_main.pkl", "wb") as f:
        pickle.dump(final_model, f)

    # ── Quantile models ───────────────────────────────────────────────────────
    qmodels = {}
    for q in [0.1, 0.5, 0.9]:
        print(f"\nTraining quantile q={q}...")
        qm = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=final_rounds,
            learning_rate=best_params.get("learning_rate", 0.03)
                          if best_params else 0.03,
            num_leaves=best_params.get("num_leaves", 63)
                       if best_params else 63,
            min_child_samples=best_params.get("min_child_samples", 100)
                              if best_params else 100,
            random_state=SEED,
        )
        qm.fit(X_tv, y_tv,
               eval_set=[(Xva, yva)],
               callbacks=[lgb.early_stopping(50, verbose=False)])
        qmodels[q] = qm

    q_preds = {q: restore(eta_te, qmodels[q].predict(Xte)) for q in [0.1, 0.5, 0.9]}
    lo, mid, hi = np.sort(
        np.stack([q_preds[0.1], q_preds[0.5], q_preds[0.9]], axis=1), axis=1
    ).T

    coverage = ((actual_te >= lo) & (actual_te <= hi)).mean() * 100
    print(f"\nQuantile coverage [q10, q90]: {coverage:.1f}%  (target ~80%)")

    q50_m = compute_metrics(actual_te, mid)
    all_metrics["mix Quantile q50"] = q50_m
    print_metrics("mix Quantile q50 (test)", q50_m)

    with open(MODELS_DIR / "qmodels.pkl", "wb") as f:
        pickle.dump(qmodels, f)

    # Feature importance (gain)
    fi_gain  = final_model.feature_importance(importance_type="gain")
    fi_split = final_model.feature_importance(importance_type="split")
    fi_names = final_model.feature_name()
    fi_data  = {
        "gain":  dict(zip(fi_names, fi_gain.tolist())),
        "split": dict(zip(fi_names, fi_split.tolist())),
    }
    with open(METRICS_DIR / "feature_importance_gain.json", "w") as f:
        json.dump(fi_data, f, indent=2)

    # Summary
    with open(METRICS_DIR / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    bm = all_metrics["Baseline (driver_eta)"]
    print(f"\n{'='*65}")
    print(f"{'Model':<28} {'MAE':>7} {'RMSE':>7} {'P90':>6} {'MAE Imp':>9} {'RMSE Imp':>10}")
    for name, m in all_metrics.items():
        mi = (bm["mae"]  - m["mae"])  / bm["mae"]  * 100
        ri = (bm["rmse"] - m["rmse"]) / bm["rmse"] * 100
        print(f"{name:<28} {m['mae']:>7.1f} {m['rmse']:>7.1f} {m['p90_ae']:>6.0f} "
              f"{mi:>8.1f}% {ri:>9.1f}%")

    return final_model, qmodels, all_metrics, avail


if __name__ == "__main__":
    main()
