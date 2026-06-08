"""Feature combination search: forward, backward, ablation, exhaustive."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from eta_pipeline.features import REGISTRY, FeatureBundle


@dataclass
class SearchEntry:
    blocks: List[str]
    columns: List[str]
    val_mae: float
    delta_vs_core: float
    n_features: int
    fit_seconds: float


@dataclass
class SearchResult:
    strategy: str
    best_blocks: List[str]
    best_columns: List[str]
    cat_cols: List[str]
    leaderboard: List[SearchEntry]
    ablation: Optional[List[SearchEntry]] = None


def _block_cols(block_names: List[str]) -> List[str]:
    cols = []
    for name in block_names:
        for col in REGISTRY[name].output_cols:
            if col not in cols:
                cols.append(col)
    return cols


def _block_cat_cols(block_names: List[str], all_cat_cols: List[str]) -> List[str]:
    candidate = _block_cols(block_names)
    return [c for c in all_cat_cols if c in candidate]


def _fit_eval(
    bundles: Dict[str, FeatureBundle],
    feature_cols: List[str],
    cat_cols: List[str],
    proxy_params,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> float:
    tr = bundles["train"]
    va = bundles["val"]

    avail = [c for c in feature_cols if c in tr.X.columns]
    if not avail:
        return float("inf")

    Xtr = tr.X[avail]
    ytr = tr.y
    if sample_frac < 1.0:
        n = max(100, int(len(Xtr) * sample_frac))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), size=n, replace=False)
        Xtr = Xtr.iloc[idx]
        ytr = ytr[idx]

    Xva = va.X[avail]
    yva = va.y

    eff_cat = [c for c in cat_cols if c in avail]
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "learning_rate": proxy_params.learning_rate,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "seed": seed,
    }
    dtrain = lgb.Dataset(Xtr, ytr, categorical_feature=eff_cat or "auto", free_raw_data=False)
    dval   = lgb.Dataset(Xva, yva, categorical_feature=eff_cat or "auto",
                         reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params, dtrain,
        num_boost_round=proxy_params.num_boost_round,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(proxy_params.early_stopping, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    pred = model.predict(Xva, num_iteration=model.best_iteration)
    return float(np.mean(np.abs(pred - yva)))


_cache: Dict[Tuple, float] = {}


def _cached_eval(
    bundles, feature_cols, cat_cols, proxy_params, sample_frac, seed
) -> float:
    key = (tuple(sorted(feature_cols)), sample_frac, seed)
    if key not in _cache:
        _cache[key] = _fit_eval(bundles, feature_cols, cat_cols, proxy_params, sample_frac, seed)
    return _cache[key]


def _all_cols(bundles: Dict[str, FeatureBundle]) -> List[str]:
    return list(bundles["train"].X.columns)


def run_forward(
    bundles: Dict[str, FeatureBundle],
    core_blocks: List[str],
    candidate_blocks: List[str],
    cfg,
) -> SearchResult:
    sr = cfg.search
    pp = sr.proxy_params
    seed = cfg.seed
    all_cat = bundles["train"].cat_cols

    current = list(core_blocks)
    remaining = list(candidate_blocks)
    core_cols = _block_cols(current)
    core_mae = _cached_eval(bundles, core_cols, _block_cat_cols(current, all_cat),
                            pp, sr.search_sample_frac, seed)
    print(f"  Core {current}: val MAE = {core_mae:.2f}s")

    leaderboard: List[SearchEntry] = [SearchEntry(
        blocks=list(current), columns=core_cols,
        val_mae=core_mae, delta_vs_core=0.0,
        n_features=len(core_cols), fit_seconds=0.0,
    )]

    while remaining and len(current) < sr.max_blocks:
        best_gain, best_block, best_mae, best_cols = -np.inf, None, core_mae, None

        for blk in remaining:
            candidate = current + [blk]
            cols = _block_cols(candidate)
            cat  = _block_cat_cols(candidate, all_cat)
            t0   = time.time()
            mae  = _cached_eval(bundles, cols, cat, pp, sr.search_sample_frac, seed)
            gain = leaderboard[-1].val_mae - mae

            entry = SearchEntry(
                blocks=candidate, columns=cols, val_mae=mae,
                delta_vs_core=core_mae - mae, n_features=len(cols),
                fit_seconds=time.time() - t0,
            )
            leaderboard.append(entry)
            print(f"    +{blk:30s} → val MAE {mae:.2f}s  (Δ={gain:+.2f}s)")

            if gain > best_gain:
                best_gain, best_block, best_mae, best_cols = gain, blk, mae, cols

        if best_gain > sr.min_gain_seconds:
            current.append(best_block)
            remaining.remove(best_block)
            print(f"  ✓ Added {best_block} (gain {best_gain:.2f}s). "
                  f"Current set: {current}")
        else:
            print(f"  No block improved by >{sr.min_gain_seconds}s. Stopping.")
            break

    best_cols = _block_cols(current)
    return SearchResult(
        strategy="forward",
        best_blocks=current,
        best_columns=best_cols,
        cat_cols=_block_cat_cols(current, all_cat),
        leaderboard=sorted(leaderboard, key=lambda e: e.val_mae),
    )


def run_ablation(
    bundles: Dict[str, FeatureBundle],
    block_names: List[str],
    cfg,
) -> List[SearchEntry]:
    sr = cfg.search
    pp = sr.proxy_params
    seed = cfg.seed
    all_cat = bundles["train"].cat_cols

    full_cols = _block_cols(block_names)
    full_cat  = _block_cat_cols(block_names, all_cat)
    full_mae  = _cached_eval(bundles, full_cols, full_cat, pp, sr.search_sample_frac, seed)
    print(f"  Full set val MAE = {full_mae:.2f}s")

    entries = []
    for blk in block_names:
        reduced = [b for b in block_names if b != blk]
        cols = _block_cols(reduced)
        cat  = _block_cat_cols(reduced, all_cat)
        t0   = time.time()
        mae  = _cached_eval(bundles, cols, cat, pp, sr.search_sample_frac, seed)
        delta = mae - full_mae
        print(f"    -{blk:30s} → val MAE {mae:.2f}s (Δ={delta:+.2f}s)")
        entries.append(SearchEntry(
            blocks=reduced, columns=cols, val_mae=mae,
            delta_vs_core=full_mae - mae, n_features=len(cols),
            fit_seconds=time.time() - t0,
        ))
    return sorted(entries, key=lambda e: e.val_mae)


def run_exhaustive(
    bundles: Dict[str, FeatureBundle],
    core_blocks: List[str],
    candidate_blocks: List[str],
    cfg,
) -> SearchResult:
    sr = cfg.search
    if len(candidate_blocks) > sr.exhaustive_max:
        raise ValueError(
            f"exhaustive search requested but {len(candidate_blocks)} candidate blocks > "
            f"exhaustive_max={sr.exhaustive_max}. Reduce candidates or switch strategy."
        )
    all_cat = bundles["train"].cat_cols
    pp = sr.proxy_params
    seed = cfg.seed

    leaderboard: List[SearchEntry] = []
    for r in range(len(candidate_blocks) + 1):
        for combo in itertools.combinations(candidate_blocks, r):
            blocks = core_blocks + list(combo)
            cols = _block_cols(blocks)
            cat  = _block_cat_cols(blocks, all_cat)
            t0   = time.time()
            mae  = _cached_eval(bundles, cols, cat, pp, sr.search_sample_frac, seed)
            leaderboard.append(SearchEntry(
                blocks=blocks, columns=cols, val_mae=mae,
                delta_vs_core=0.0, n_features=len(cols),
                fit_seconds=time.time() - t0,
            ))

    leaderboard.sort(key=lambda e: e.val_mae)
    best = leaderboard[0]
    core_mae = _cached_eval(
        bundles, _block_cols(core_blocks), _block_cat_cols(core_blocks, all_cat),
        pp, sr.search_sample_frac, seed,
    )
    for e in leaderboard:
        e.delta_vs_core = core_mae - e.val_mae

    return SearchResult(
        strategy="exhaustive",
        best_blocks=best.blocks,
        best_columns=best.columns,
        cat_cols=_block_cat_cols(best.blocks, all_cat),
        leaderboard=leaderboard,
    )


def run_search(
    bundles: Dict[str, FeatureBundle],
    core_blocks: List[str],
    candidate_blocks: List[str],
    cfg,
) -> SearchResult:
    _cache.clear()
    strategy = cfg.search.strategy
    print(f"\n=== Feature search (strategy={strategy}) ===")

    if strategy == "forward":
        result = run_forward(bundles, core_blocks, candidate_blocks, cfg)
    elif strategy == "exhaustive":
        result = run_exhaustive(bundles, core_blocks, candidate_blocks, cfg)
    elif strategy == "ablation":
        entries = run_ablation(bundles, core_blocks + candidate_blocks, cfg)
        all_cat = bundles["train"].cat_cols
        full = core_blocks + candidate_blocks
        result = SearchResult(
            strategy="ablation",
            best_blocks=full,
            best_columns=_block_cols(full),
            cat_cols=_block_cat_cols(full, all_cat),
            leaderboard=entries,
        )
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")

    # Optionally run ablation on top of forward/exhaustive
    if cfg.search.also_run_ablation and strategy != "ablation":
        print("\n  Running ablation on best block set...")
        result.ablation = run_ablation(bundles, result.best_blocks, cfg)

    return result
