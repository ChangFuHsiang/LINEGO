"""Config dataclasses + YAML loader + validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class DataConfig:
    path: str = "data/raw/trip_stats_eta.parquet"
    clip_target_quantiles: Optional[List[float]] = None

    def __post_init__(self):
        if self.clip_target_quantiles is not None:
            self.clip_target_quantiles = list(self.clip_target_quantiles)


@dataclass
class SplitConfig:
    train_months: List[int] = field(default_factory=lambda: [12, 1, 2, 3])
    val_months: List[int] = field(default_factory=lambda: [4])
    test_months: List[int] = field(default_factory=lambda: [5])
    lny_window: List[str] = field(default_factory=lambda: ["2026-01-28", "2026-02-05"])
    tz_offset_hours: int = 8


@dataclass
class FeatureConfig:
    core_blocks: List[str] = field(default_factory=lambda: ["base_eta"])
    candidate_blocks: List[str] = field(default_factory=list)
    target_enc_smoothing: int = 100
    target_enc_folds: int = 5


@dataclass
class ProxyParams:
    learning_rate: float = 0.05
    num_boost_round: int = 400
    early_stopping: int = 50


@dataclass
class SearchConfig:
    mode: str = "sequential"
    strategy: str = "forward"
    min_gain_seconds: float = 0.2
    max_blocks: int = 12
    exhaustive_max: int = 6
    search_sample_frac: float = 1.0
    proxy_params: ProxyParams = field(default_factory=ProxyParams)
    also_run_ablation: bool = True


@dataclass
class HParamRange:
    low: float = 0.0
    high: float = 1.0
    log: bool = False


@dataclass
class TuningSpace:
    learning_rate: HParamRange = field(default_factory=lambda: HParamRange(0.01, 0.1, True))
    num_leaves: HParamRange = field(default_factory=lambda: HParamRange(32, 256))
    min_data_in_leaf: HParamRange = field(default_factory=lambda: HParamRange(50, 500))
    feature_fraction: HParamRange = field(default_factory=lambda: HParamRange(0.6, 1.0))
    bagging_fraction: HParamRange = field(default_factory=lambda: HParamRange(0.6, 1.0))
    bagging_freq: HParamRange = field(default_factory=lambda: HParamRange(1, 10))
    lambda_l1: HParamRange = field(default_factory=lambda: HParamRange(1e-4, 10.0, True))
    lambda_l2: HParamRange = field(default_factory=lambda: HParamRange(1e-4, 10.0, True))


@dataclass
class TuningConfig:
    n_trials: int = 150
    timeout_seconds: int = 7200
    num_boost_round: int = 2000
    early_stopping: int = 50
    round_inflation: float = 1.1
    pruner: str = "median"
    n_startup_trials: int = 10
    space: TuningSpace = field(default_factory=TuningSpace)


@dataclass
class ModelConfig:
    objective: str = "regression_l1"
    metric: str = "mae"
    quantile_alpha: Optional[float] = None


@dataclass
class ReportConfig:
    dir: str = "reports"
    emit_json_sidecar: bool = True
    emit_plots: bool = False
    min_segment_rows: int = 200
    distance_buckets_km: List[float] = field(default_factory=lambda: [0, 2, 5, 10, 20])
    driver_eta_buckets_s: List[float] = field(default_factory=lambda: [0, 120, 300, 600])


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "cuda"
    phases: List[str] = field(default_factory=lambda: ["feature", "tune", "final"])
    run_id: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def __post_init__(self):
        if self.run_id is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            cfg_hash = hashlib.md5(
                json.dumps(asdict(self), sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            self.run_id = f"{ts}-{cfg_hash}"

    @property
    def all_blocks(self) -> List[str]:
        return list(self.features.core_blocks) + list(self.features.candidate_blocks)


# ── helpers ──────────────────────────────────────────────────────────────────

def _from_dict(cls, d: dict):
    """Recursively build a dataclass from a (possibly nested) dict."""
    if d is None:
        return cls()
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for name, fld in fields.items():
        if name not in d:
            continue
        val = d[name]
        ft = fld.type
        # Resolve forward references / string annotations
        if isinstance(ft, str):
            import sys
            ft = eval(ft, sys.modules[cls.__module__].__dict__)
        # Check if ft is a dataclass
        import dataclasses
        if dataclasses.is_dataclass(ft) and isinstance(val, dict):
            val = _from_dict(ft, val)
        kwargs[name] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = RunConfig(
        seed=raw.get("run", {}).get("seed", 42),
        device=raw.get("run", {}).get("device", "cuda"),
        phases=raw.get("run", {}).get("phases", ["feature", "tune", "final"]),
    )

    if "data" in raw:
        d = raw["data"]
        cfg.data = DataConfig(
            path=d.get("path", cfg.data.path),
            clip_target_quantiles=d.get("clip_target_quantiles"),
        )

    if "split" in raw:
        s = raw["split"]
        cfg.split = SplitConfig(
            train_months=list(s.get("train_months", cfg.split.train_months)),
            val_months=list(s.get("val_months", cfg.split.val_months)),
            test_months=list(s.get("test_months", cfg.split.test_months)),
            lny_window=list(s.get("lny_window", cfg.split.lny_window)),
            tz_offset_hours=s.get("tz_offset_hours", cfg.split.tz_offset_hours),
        )

    if "features" in raw:
        fe = raw["features"]
        cfg.features = FeatureConfig(
            core_blocks=list(fe.get("core_blocks", cfg.features.core_blocks)),
            candidate_blocks=list(fe.get("candidate_blocks", [])),
            target_enc_smoothing=fe.get("target_enc_smoothing", 100),
            target_enc_folds=fe.get("target_enc_folds", 5),
        )

    if "search" in raw:
        sr = raw["search"]
        pp_raw = sr.get("proxy_params", {})
        cfg.search = SearchConfig(
            mode=sr.get("mode", "sequential"),
            strategy=sr.get("strategy", "forward"),
            min_gain_seconds=sr.get("min_gain_seconds", 0.2),
            max_blocks=sr.get("max_blocks", 12),
            exhaustive_max=sr.get("exhaustive_max", 6),
            search_sample_frac=sr.get("search_sample_frac", 1.0),
            proxy_params=ProxyParams(
                learning_rate=pp_raw.get("learning_rate", 0.05),
                num_boost_round=pp_raw.get("num_boost_round", 400),
                early_stopping=pp_raw.get("early_stopping", 50),
            ),
            also_run_ablation=sr.get("also_run_ablation", True),
        )

    if "tuning" in raw:
        tu = raw["tuning"]
        sp = tu.get("space", {})

        def _hp(name, default_low, default_high, default_log=False):
            d = sp.get(name, {})
            return HParamRange(
                low=d.get("low", default_low),
                high=d.get("high", default_high),
                log=d.get("log", default_log),
            )

        cfg.tuning = TuningConfig(
            n_trials=tu.get("n_trials", 150),
            timeout_seconds=tu.get("timeout_seconds", 7200),
            num_boost_round=tu.get("num_boost_round", 2000),
            early_stopping=tu.get("early_stopping", 50),
            round_inflation=tu.get("round_inflation", 1.1),
            pruner=tu.get("pruner", "median"),
            n_startup_trials=tu.get("n_startup_trials", 10),
            space=TuningSpace(
                learning_rate=_hp("learning_rate", 0.01, 0.1, True),
                num_leaves=_hp("num_leaves", 32, 256),
                min_data_in_leaf=_hp("min_data_in_leaf", 50, 500),
                feature_fraction=_hp("feature_fraction", 0.6, 1.0),
                bagging_fraction=_hp("bagging_fraction", 0.6, 1.0),
                bagging_freq=_hp("bagging_freq", 1, 10),
                lambda_l1=_hp("lambda_l1", 1e-4, 10.0, True),
                lambda_l2=_hp("lambda_l2", 1e-4, 10.0, True),
            ),
        )

    if "model" in raw:
        mo = raw["model"]
        cfg.model = ModelConfig(
            objective=mo.get("objective", "regression_l1"),
            metric=mo.get("metric", "mae"),
            quantile_alpha=mo.get("quantile_alpha"),
        )

    if "report" in raw:
        re = raw["report"]
        cfg.report = ReportConfig(
            dir=re.get("dir", "reports"),
            emit_json_sidecar=re.get("emit_json_sidecar", True),
            emit_plots=re.get("emit_plots", False),
            min_segment_rows=re.get("min_segment_rows", 200),
            distance_buckets_km=list(re.get("distance_buckets_km", [0, 2, 5, 10, 20])),
            driver_eta_buckets_s=list(re.get("driver_eta_buckets_s", [0, 120, 300, 600])),
        )

    _validate(cfg, path)
    return cfg


def _validate(cfg: RunConfig, source: Any = None):
    from eta_pipeline.features import REGISTRY
    known = set(REGISTRY.keys())
    for b in cfg.all_blocks:
        if b not in known:
            raise ValueError(f"Unknown block '{b}' in config (from {source}). Known: {sorted(known)}")

    months = set(cfg.split.train_months) | set(cfg.split.val_months) | set(cfg.split.test_months)
    overlaps = (
        set(cfg.split.train_months) & set(cfg.split.val_months),
        set(cfg.split.train_months) & set(cfg.split.test_months),
        set(cfg.split.val_months) & set(cfg.split.test_months),
    )
    for o in overlaps:
        if o:
            raise ValueError(f"Split months overlap: {o}")

    valid_phases = {"feature", "tune", "final"}
    for p in cfg.phases:
        if p not in valid_phases:
            raise ValueError(f"Unknown phase '{p}'. Valid: {valid_phases}")
