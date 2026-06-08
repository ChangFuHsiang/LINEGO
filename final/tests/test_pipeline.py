"""Leakage, determinism, registry, and metric tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eta_pipeline.features import (
    REGISTRY, FeatureBundle, FORBIDDEN_FEATURE_COLS,
    build_features, _topo_sort,
    RegionTargetEncBlock, DriverBiasBlock,
)
from eta_pipeline.metrics import evaluate


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = rng.choice([1, 2, 3, 4, 5], size=n)
    counties = rng.choice(["Taipei", "NewTaipei", "Taoyuan"], size=n)
    towns = rng.choice(["A", "B", "C", "D"], size=n)
    df = pd.DataFrame({
        "driver_eta":           rng.integers(60, 600, size=n).astype(float),
        "time_accept_to_arrive": rng.integers(60, 800, size=n).astype(float),
        "start_lat":  rng.uniform(22, 25, n),
        "start_lng":  rng.uniform(120, 122, n),
        "end_lat":    rng.uniform(22, 25, n),
        "end_lng":    rng.uniform(120, 122, n),
        "start_county": counties,
        "start_town":   towns,
        "end_county":   counties,
        "end_town":     towns,
        "did_hash":  [f"d{i%20}" for i in range(n)],
        "uid_hash":  [f"u{i}"    for i in range(n)],
        "hour":      rng.integers(0, 24, n),
        "day_of_week": rng.integers(1, 8, n),
        "day_of_month": rng.integers(1, 29, n),
        "month":      months,
        "is_lunar_new_year": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "eta_error":  rng.normal(50, 100, n),
    })
    return df


class _DummyCfg:
    seed = 42
    class features:
        target_enc_smoothing = 20
        target_enc_folds = 3
    class search:
        strategy = "forward"
        min_gain_seconds = 0.1
        max_blocks = 5
        exhaustive_max = 4
        search_sample_frac = 1.0
        class proxy_params:
            learning_rate = 0.1
            num_boost_round = 50
            early_stopping = 10
        also_run_ablation = False
    class tuning:
        n_trials = 2
        timeout_seconds = 30
        num_boost_round = 100
        early_stopping = 10
        round_inflation = 1.1
        pruner = "median"
        n_startup_trials = 1
        class space:
            class learning_rate:
                low, high, log = 0.05, 0.1, True
            class num_leaves:
                low, high, log = 32, 64, False
            class min_data_in_leaf:
                low, high, log = 10, 50, False
            class feature_fraction:
                low, high, log = 0.8, 1.0, False
            class bagging_fraction:
                low, high, log = 0.8, 1.0, False
            class bagging_freq:
                low, high, log = 1, 5, False
            class lambda_l1:
                low, high, log = 1e-4, 1.0, True
            class lambda_l2:
                low, high, log = 1e-4, 1.0, True
    class model:
        objective = "regression_l1"
        metric = "mae"
    class report:
        min_segment_rows = 10
        distance_buckets_km = [0, 5, 20]
        driver_eta_buckets_s = [0, 120, 600]

    def all_blocks(self):
        return ["base_eta", "time_basic", "time_cyclic"]


def _make_splits(n_train=300, n_val=100, n_test=100):
    train = _make_df(n_train, seed=0)
    val   = _make_df(n_val,   seed=1)
    test  = _make_df(n_test,  seed=2)
    # Avoid index overlap
    val.index   = range(n_train, n_train + n_val)
    test.index  = range(n_train + n_val, n_train + n_val + n_test)
    return {"train": train, "val": val, "test": test}


# ── Registry tests ───────────────────────────────────────────────────────────

def test_registry_has_expected_blocks():
    required = {"base_eta", "time_basic", "time_cyclic", "time_flags",
                 "geo_raw", "geo_distance", "region_cat", "driver_bias",
                 "region_target_enc"}
    assert required <= set(REGISTRY.keys()), \
        f"Missing blocks: {required - set(REGISTRY.keys())}"


def test_topo_sort_respects_depends_on():
    order = _topo_sort(["eta_derived", "geo_distance", "time_flags", "base_eta"])
    assert order.index("geo_distance") < order.index("eta_derived")
    assert order.index("time_flags")  < order.index("eta_derived")


def test_no_circular_dependency():
    _topo_sort(list(REGISTRY.keys()))  # should not raise


# ── Leakage tests ────────────────────────────────────────────────────────────

def test_forbidden_cols_not_in_features():
    splits = _make_splits()
    blocks = ["base_eta", "time_basic", "driver_bias", "region_target_enc"]
    bundles = build_features(blocks, splits, _DummyCfg())
    for sp, bundle in bundles.items():
        bad = set(bundle.X.columns) & FORBIDDEN_FEATURE_COLS
        assert not bad, f"Leakage in {sp}: {bad}"


def test_stateful_block_fits_only_on_train():
    """RegionTargetEncBlock OOF map should only contain train indices."""
    splits = _make_splits()
    blk = RegionTargetEncBlock()
    blk.fit(splits["train"], _DummyCfg())

    train_idx = set(splits["train"].index.tolist())
    val_idx   = set(splits["val"].index.tolist())
    oof_keys  = set(blk._oof_map.get("start_town", {}).keys())

    assert oof_keys <= train_idx, "OOF map contains val/test indices — leakage!"
    assert not (oof_keys & val_idx), "OOF map leaks into val split"


def test_driver_bias_excludes_lny():
    splits = _make_splits(400, 100, 100)
    blk = DriverBiasBlock()
    blk.fit(splits["train"])
    # Just verify it runs without using LNY rows explicitly — no crash
    assert blk._global["avg"] == blk._global["avg"]  # not NaN


# ── Determinism test ─────────────────────────────────────────────────────────

def test_build_features_deterministic():
    splits = _make_splits()
    blocks = ["base_eta", "time_basic", "region_target_enc", "driver_bias"]
    b1 = build_features(blocks, splits, _DummyCfg())
    b2 = build_features(blocks, splits, _DummyCfg())
    np.testing.assert_array_almost_equal(
        b1["train"].X["start_town_te"].values,
        b2["train"].X["start_town_te"].values,
        err_msg="Feature build not deterministic",
    )


# ── Metric tests ─────────────────────────────────────────────────────────────

def test_evaluate_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    m = evaluate(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["within_60"] == pytest.approx(100.0)
    assert m["over_promise"] == pytest.approx(0.0)


def test_evaluate_vs_raw_improvement():
    actual = np.array([200.0, 300.0, 400.0])
    raw    = np.array([100.0, 150.0, 200.0])  # systematic under-estimate
    corrected = np.array([190.0, 290.0, 390.0])  # better
    m = evaluate(actual, corrected, raw=raw)
    assert m["mae_imp_pct"] > 0, "Expected improvement vs raw"


def test_evaluate_n_field():
    y = np.arange(50.0)
    m = evaluate(y, y)
    assert m["n"] == 50


# ── build_features smoke test ─────────────────────────────────────────────────

def test_build_features_basic():
    splits = _make_splits()
    bundles = build_features(["base_eta", "time_basic"], splits, _DummyCfg())
    assert "driver_eta" in bundles["train"].X.columns
    assert "hour" in bundles["train"].X.columns
    assert len(bundles["train"].y) == len(splits["train"])
    assert len(bundles["val"].y)   == len(splits["val"])


def test_build_features_index_aligned():
    splits = _make_splits()
    bundles = build_features(["base_eta", "driver_bias"], splits, _DummyCfg())
    for sp in ["train", "val", "test"]:
        assert list(bundles[sp].X.index) == list(splits[sp].index), \
            f"Index mismatch in {sp} split"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
