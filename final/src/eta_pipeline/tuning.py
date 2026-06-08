"""Optuna hyperparameter search — resumable, SQLite-persisted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import optuna
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

# LightGBMPruningCallback moved to optuna-integration in optuna 4.x
LightGBMPruningCallback = None
try:
    from optuna_integration import LightGBMPruningCallback
except ImportError:
    try:
        from optuna.integration import LightGBMPruningCallback
    except ImportError:
        pass  # pruner will still work, just without per-step pruning

from eta_pipeline.features import FeatureBundle


@dataclass
class TuningResult:
    best_params: dict
    best_iteration: int
    val_mae: float
    n_trials: int
    param_importances: Optional[Dict[str, float]] = None


def _build_params(trial, cfg, device: str) -> dict:
    sp = cfg.tuning.space

    def suggest(name, hpr, is_int=False):
        if is_int:
            return trial.suggest_int(name, int(hpr.low), int(hpr.high))
        return trial.suggest_float(name, hpr.low, hpr.high, log=hpr.log)

    return {
        "objective":         cfg.model.objective,
        "metric":            cfg.model.metric,
        "verbosity":         -1,
        "device":            device,
        "learning_rate":     suggest("learning_rate",     sp.learning_rate),
        "num_leaves":        suggest("num_leaves",        sp.num_leaves,        is_int=True),
        "min_data_in_leaf":  suggest("min_data_in_leaf",  sp.min_data_in_leaf,  is_int=True),
        "feature_fraction":  suggest("feature_fraction",  sp.feature_fraction),
        "bagging_fraction":  suggest("bagging_fraction",  sp.bagging_fraction),
        "bagging_freq":      suggest("bagging_freq",      sp.bagging_freq,      is_int=True),
        "lambda_l1":         suggest("lambda_l1",         sp.lambda_l1),
        "lambda_l2":         suggest("lambda_l2",         sp.lambda_l2),
        "seed":              cfg.seed,
    }


def run_tuning(
    bundles: Dict[str, FeatureBundle],
    feature_cols: List[str],
    cat_cols: List[str],
    cfg,
    artifacts_dir: Path,
    run_id: str,
    device: str = "cpu",
) -> TuningResult:
    if not OPTUNA_OK:
        raise ImportError("optuna is not installed. Run: pip install optuna")

    tr = bundles["train"]
    va = bundles["val"]

    avail = [c for c in feature_cols if c in tr.X.columns]
    eff_cat = [c for c in cat_cols if c in avail]

    Xtr = tr.X[avail]
    ytr = tr.y
    Xva = va.X[avail]
    yva = va.y

    dtrain = lgb.Dataset(Xtr, ytr, categorical_feature=eff_cat or "auto", free_raw_data=False)
    dval   = lgb.Dataset(Xva, yva, categorical_feature=eff_cat or "auto",
                         reference=dtrain, free_raw_data=False)

    tu = cfg.tuning
    storage = f"sqlite:///{artifacts_dir / f'study_{run_id}.db'}"

    pruner_cls = (
        optuna.pruners.MedianPruner(n_startup_trials=tu.n_startup_trials)
        if tu.pruner == "median"
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name=run_id,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=pruner_cls,
    )

    def objective(trial):
        params = _build_params(trial, cfg, device)
        cbs = [
            lgb.early_stopping(tu.early_stopping, verbose=False),
            lgb.log_evaluation(-1),
        ]
        if LightGBMPruningCallback is not None:
            cbs.append(LightGBMPruningCallback(trial, "l1", valid_name="valid_0"))

        try:
            model = lgb.train(
                params, dtrain,
                num_boost_round=tu.num_boost_round,
                valid_sets=[dval],
                callbacks=cbs,
            )
        except optuna.exceptions.TrialPruned:
            raise

        pred = model.predict(Xva, num_iteration=model.best_iteration)
        mae  = float(np.mean(np.abs(pred - yva)))
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=tu.n_trials,
        timeout=tu.timeout_seconds,
        show_progress_bar=False,
    )

    best = study.best_trial
    best_iter = int(best.user_attrs.get("best_iteration", tu.num_boost_round))

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = None

    print(f"\nTuning done — best val MAE: {best.value:.2f}s  "
          f"(best_iteration={best_iter})  "
          f"trials: {len(study.trials)}")
    print(f"  Best params: {best.params}")

    final_params = {
        "objective":        cfg.model.objective,
        "metric":           cfg.model.metric,
        "verbosity":        -1,
        "device":           device,
        "seed":             cfg.seed,
        **best.params,
    }
    return TuningResult(
        best_params=final_params,
        best_iteration=best_iter,
        val_mae=float(best.value),
        n_trials=len(study.trials),
        param_importances=importances,
    )
