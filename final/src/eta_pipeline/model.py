"""LightGBM train/predict/save wrappers with CUDA detection."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd


def detect_device(requested: str) -> Tuple[str, str]:
    """Try requested device; return (actual, message)."""
    if requested == "cpu":
        return "cpu", "cpu (requested)"

    try:
        tiny_X = pd.DataFrame({"x": np.zeros(10), "y": np.zeros(10)})
        tiny_y = np.zeros(10)
        ds = lgb.Dataset(tiny_X, tiny_y, free_raw_data=False)
        lgb.train(
            {"objective": "regression", "device": "cuda", "num_boost_round": 1,
             "verbosity": -1},
            ds, num_boost_round=1,
        )
        return "cuda", f"cuda (requested: {requested})"
    except Exception as e:
        return "cpu", f"cpu (fallback from {requested}: {e})"


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    cat_features: List[str],
    num_boost_round: int,
    early_stopping: int,
    feature_cols: Optional[List[str]] = None,
) -> lgb.Booster:
    if feature_cols:
        X_train = X_train[feature_cols]
        X_val   = X_val[feature_cols]

    effective_cat = [c for c in cat_features if c in X_train.columns]

    dtrain = lgb.Dataset(X_train, label=y_train,
                         categorical_feature=effective_cat or "auto",
                         free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val,
                         categorical_feature=effective_cat or "auto",
                         reference=dtrain, free_raw_data=False)

    callbacks = [lgb.early_stopping(early_stopping, verbose=False),
                 lgb.log_evaluation(100)]

    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def predict(model: lgb.Booster, X: pd.DataFrame,
            feature_cols: Optional[List[str]] = None) -> np.ndarray:
    if feature_cols:
        X = X[feature_cols]
    return model.predict(X, num_iteration=model.best_iteration)


def restore_eta(driver_eta: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """corrected_eta = driver_eta + pred  (additive residual)."""
    return np.maximum(0.0, driver_eta + pred)


def feature_importance(model: lgb.Booster, imp_type: str = "gain") -> pd.DataFrame:
    names = model.feature_name()
    vals  = model.feature_importance(importance_type=imp_type)
    return pd.DataFrame({"feature": names, "importance": vals}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


def save_model(model: lgb.Booster, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> lgb.Booster:
    with open(path, "rb") as f:
        return pickle.load(f)
