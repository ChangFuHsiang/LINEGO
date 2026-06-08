"""Feature block registry with leakage-safe fit/transform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ── Leakage deny-list ─────────────────────────────────────────────────────────
FORBIDDEN_FEATURE_COLS = {
    "time_accept_to_arrive", "time_start_to_finish",
    "trip_id", "uid_hash", "end_address", "eta_error",
}

# ── FeatureBundle ─────────────────────────────────────────────────────────────
@dataclass
class FeatureBundle:
    X: pd.DataFrame        # all superset columns; index aligned with raw split
    y: np.ndarray          # eta_error (target)
    cat_cols: List[str]    # categorical columns (subset of X.columns)


# ── Base class ────────────────────────────────────────────────────────────────
class FeatureBlock(ABC):
    name: str
    output_cols: List[str]
    categorical_cols: List[str] = []
    depends_on: List[str] = []
    cost: str = "cheap"     # "cheap" | "expensive"

    def fit(self, train_df: pd.DataFrame, cfg=None) -> None:
        """Fit on train only. No-op for stateless blocks."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with only output_cols, same index as df."""


# ── Stateless blocks ──────────────────────────────────────────────────────────

class BaseEtaBlock(FeatureBlock):
    name = "base_eta"
    output_cols = ["driver_eta"]
    categorical_cols = []
    cost = "cheap"

    def transform(self, df):
        return df[["driver_eta"]].copy()


class TimeBasicBlock(FeatureBlock):
    name = "time_basic"
    output_cols = ["hour", "day_of_week", "is_weekend", "month", "day_of_month"]
    cost = "cheap"

    def transform(self, df):
        out = df[["hour", "day_of_week", "month", "day_of_month"]].copy()
        out["is_weekend"] = (df["day_of_week"] >= 6).astype(int)
        return out[self.output_cols]


class TimeCyclicBlock(FeatureBlock):
    name = "time_cyclic"
    output_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    cost = "cheap"

    def transform(self, df):
        h = df["hour"]
        d = df["day_of_week"]
        return pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * h / 24),
            "hour_cos": np.cos(2 * np.pi * h / 24),
            "dow_sin":  np.sin(2 * np.pi * d / 7),
            "dow_cos":  np.cos(2 * np.pi * d / 7),
        }, index=df.index)


class TimeFlagsBlock(FeatureBlock):
    name = "time_flags"
    output_cols = ["is_rush_hour", "is_night", "is_lunar_new_year"]
    cost = "cheap"

    def transform(self, df):
        h = df["hour"]
        return pd.DataFrame({
            "is_rush_hour": (((h >= 7) & (h <= 9)) | ((h >= 17) & (h <= 19))).astype(int),
            "is_night":     ((h >= 22) | (h <= 5)).astype(int),
            "is_lunar_new_year": df["is_lunar_new_year"].values,
        }, index=df.index)


class GeoRawBlock(FeatureBlock):
    name = "geo_raw"
    output_cols = ["start_lat", "start_lng", "end_lat", "end_lng"]
    cost = "cheap"

    def transform(self, df):
        return df[self.output_cols].copy()


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(r2)
    y = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y))


class GeoDistanceBlock(FeatureBlock):
    name = "geo_distance"
    output_cols = ["haversine_m", "lat_delta", "lng_delta", "bearing"]
    cost = "cheap"

    def transform(self, df):
        s_lat = df["start_lat"].values
        s_lng = df["start_lng"].values
        e_lat = df["end_lat"].values
        e_lng = df["end_lng"].values
        return pd.DataFrame({
            "haversine_m": _haversine_m(s_lat, s_lng, e_lat, e_lng),
            "lat_delta":   e_lat - s_lat,
            "lng_delta":   e_lng - s_lng,
            "bearing":     _bearing(s_lat, s_lng, e_lat, e_lng),
        }, index=df.index)


_CITY_CENTERS = {
    "Taipei":     (25.0408, 121.5681),
    "NewTaipei":  (25.0170, 121.4628),
    "Taoyuan":    (24.9937, 121.3009),
    "Taichung":   (24.1435, 120.6816),
    "Tainan":     (22.9997, 120.2270),
    "Kaohsiung":  (22.6333, 120.3014),
}


def _dist_from_nearest_center(lat, lng):
    dists = np.stack([
        _haversine_m(lat, lng, clat, clng)
        for clat, clng in _CITY_CENTERS.values()
    ], axis=1)
    return dists.min(axis=1)


class GeoCenterBlock(FeatureBlock):
    name = "geo_center"
    output_cols = ["dist_from_city_center_start", "dist_from_city_center_end"]
    cost = "cheap"

    def transform(self, df):
        return pd.DataFrame({
            "dist_from_city_center_start": _dist_from_nearest_center(
                df["start_lat"].values, df["start_lng"].values),
            "dist_from_city_center_end":   _dist_from_nearest_center(
                df["end_lat"].values, df["end_lng"].values),
        }, index=df.index)


class GeoFlagsBlock(FeatureBlock):
    name = "geo_flags"
    output_cols = ["same_county", "same_town", "is_cross_county"]
    cost = "cheap"

    def transform(self, df):
        sc = (df["start_county"] == df["end_county"]).astype(int)
        st = (df["start_town"] == df["end_town"]).astype(int)
        return pd.DataFrame({
            "same_county":    sc,
            "same_town":      st,
            "is_cross_county": 1 - sc,
        }, index=df.index)


class RegionCatBlock(FeatureBlock):
    name = "region_cat"
    output_cols = ["start_county", "start_town", "end_county", "end_town"]
    categorical_cols = ["start_county", "start_town", "end_county", "end_town"]
    cost = "cheap"

    _train_categories: Dict[str, List] = {}

    def fit(self, train_df, cfg=None):
        self._train_categories = {}
        for col in self.categorical_cols:
            cats = list(train_df[col].astype(str).unique())
            if "unknown" not in cats:
                cats = ["unknown"] + cats
            self._train_categories[col] = sorted(cats)

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            if self._train_categories:
                cats = self._train_categories.get(col, None)
                out[col] = pd.Categorical(df[col].astype(str), categories=cats)
            else:
                out[col] = df[col].astype(str)
        return out


class EtaDerivedBlock(FeatureBlock):
    name = "eta_derived"
    output_cols = ["log_driver_eta", "eta_per_km", "eta_x_rush"]
    depends_on = ["geo_distance", "time_flags"]
    cost = "cheap"

    def transform(self, df):
        log_eta = np.log1p(df["driver_eta"].values)
        dist = df.get("haversine_m", pd.Series(np.zeros(len(df)), index=df.index))
        if hasattr(dist, "values"):
            dist = dist.values
        safe_dist = np.where(dist > 0, dist, np.nan)
        eta_per_km = df["driver_eta"].values / (safe_dist / 1000)

        rush = df.get("is_rush_hour", pd.Series(np.zeros(len(df)), index=df.index))
        if hasattr(rush, "values"):
            rush = rush.values
        eta_x_rush = log_eta * rush

        return pd.DataFrame({
            "log_driver_eta": log_eta,
            "eta_per_km":     eta_per_km,
            "eta_x_rush":     eta_x_rush,
        }, index=df.index)


# ── Stateful blocks ───────────────────────────────────────────────────────────

class RegionFreqBlock(FeatureBlock):
    name = "region_freq"
    output_cols = ["start_town_freq", "start_county_freq", "end_town_freq", "end_county_freq"]
    cost = "cheap"

    _freq: Dict[str, Dict] = {}

    def fit(self, train_df, cfg=None):
        self._freq = {}
        n = len(train_df)
        for col in ["start_town", "start_county", "end_town", "end_county"]:
            counts = train_df[col].astype(str).value_counts().to_dict()
            self._freq[col] = {k: v / n for k, v in counts.items()}

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        pairs = [
            ("start_town",   "start_town_freq"),
            ("start_county", "start_county_freq"),
            ("end_town",     "end_town_freq"),
            ("end_county",   "end_county_freq"),
        ]
        for src, dst in pairs:
            freq = self._freq.get(src, {})
            out[dst] = df[src].astype(str).map(freq).fillna(0.0)
        return out


def _smoothed_agg(train: pd.DataFrame, key: str, target: str, m: float) -> Dict:
    g = float(train[target].mean())
    agg = train.groupby(key)[target].agg(["mean", "count"]).reset_index()
    agg["te"] = (agg["count"] * agg["mean"] + m * g) / (agg["count"] + m)
    return dict(zip(agg[key], agg["te"])), g


def _oof_target_enc(train: pd.DataFrame, key: str, target: str, m: float,
                    n_folds: int, seed: int) -> Dict[Any, float]:
    """Return dict trip_id → OOF TE value for train rows."""
    n = len(train)
    g = float(train[target].mean())
    te_vals = np.full(n, g)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx = np.arange(n)

    for held_idx, oof_idx in kf.split(idx):
        other = train.iloc[held_idx]
        fold_rows = train.iloc[oof_idx]
        lookup, _ = _smoothed_agg(other, key, target, m)
        te_vals[oof_idx] = [lookup.get(k, g) for k in fold_rows[key].astype(str).tolist()]

    return dict(zip(train.index.tolist(), te_vals))


class RegionTargetEncBlock(FeatureBlock):
    name = "region_target_enc"
    output_cols = ["start_town_te", "end_town_te"]
    cost = "expensive"

    _lookup: Dict[str, Dict] = {}
    _global: Dict[str, float] = {}
    _oof_map: Dict[str, Dict] = {}

    def fit(self, train_df, cfg=None):
        m = cfg.features.target_enc_smoothing if cfg else 100
        n_folds = cfg.features.target_enc_folds if cfg else 5
        seed = cfg.seed if cfg else 42

        for col in ["start_town", "end_town"]:
            train_str = train_df.copy()
            train_str[col] = train_str[col].astype(str)
            lookup, g = _smoothed_agg(train_str, col, "eta_error", m)
            self._lookup[col] = lookup
            self._global[col] = g
            oof = _oof_target_enc(train_str, col, "eta_error", m, n_folds, seed)
            self._oof_map[col] = oof

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for col, out_col in [("start_town", "start_town_te"), ("end_town", "end_town_te")]:
            g = self._global.get(col, 0.0)
            lookup = self._lookup.get(col, {})
            oof_map = self._oof_map.get(col, {})

            if oof_map:
                # Use OOF values for rows that have them (train), global lookup for others (val/test)
                vals = [
                    oof_map[idx] if idx in oof_map else lookup.get(str(k), g)
                    for idx, k in zip(df.index.tolist(), df[col].astype(str).tolist())
                ]
            else:
                vals = [lookup.get(str(k), g) for k in df[col].astype(str).tolist()]
            out[out_col] = vals
        return out


class DriverBiasBlock(FeatureBlock):
    name = "driver_bias"
    output_cols = ["driver_avg_error", "driver_median_error", "driver_error_std"]
    cost = "expensive"

    _stats: pd.DataFrame = None
    _global: Dict[str, float] = {}

    def fit(self, train_df, cfg=None):
        # Exclude LNY rows from driver aggregation (regime outlier)
        no_lny = train_df[train_df["is_lunar_new_year"] == 0]
        g_mean   = float(no_lny["eta_error"].mean())
        g_median = float(no_lny["eta_error"].median())
        g_std    = float(no_lny["eta_error"].std())
        self._global = {"avg": g_mean, "median": g_median, "std": g_std}

        m = 30  # smoothing prior weight
        grp = no_lny.groupby("did_hash")["eta_error"].agg(
            n="count", mean="mean", median="median", std="std"
        ).reset_index()
        grp["driver_avg_error"] = (grp["n"] * grp["mean"] + m * g_mean) / (grp["n"] + m)
        grp["driver_median_error"] = (grp["n"] * grp["median"] + m * g_median) / (grp["n"] + m)
        grp["driver_error_std"] = grp["std"].fillna(g_std)
        self._stats = grp.set_index("did_hash")[
            ["driver_avg_error", "driver_median_error", "driver_error_std"]
        ].to_dict("index")

    def transform(self, df):
        g = self._global
        rows = []
        for did in df["did_hash"].tolist():
            if self._stats and did in self._stats:
                rows.append(self._stats[did])
            else:
                rows.append({
                    "driver_avg_error":    g.get("avg", 0.0),
                    "driver_median_error": g.get("median", 0.0),
                    "driver_error_std":    g.get("std", 0.0),
                })
        return pd.DataFrame(rows, index=df.index)


class DriverExperienceBlock(FeatureBlock):
    name = "driver_experience"
    output_cols = ["driver_trip_count"]
    cost = "cheap"

    _counts: Dict[str, int] = {}
    _global_count: float = 0.0

    def fit(self, train_df, cfg=None):
        self._counts = train_df["did_hash"].value_counts().to_dict()
        self._global_count = float(np.median(list(self._counts.values())))

    def transform(self, df):
        g = self._global_count
        vals = [self._counts.get(d, g) for d in df["did_hash"].tolist()]
        return pd.DataFrame({"driver_trip_count": vals}, index=df.index)


class DestDemandBlock(FeatureBlock):
    name = "dest_demand"
    output_cols = ["dest_demand_score"]
    cost = "cheap"

    _counts: Dict[str, int] = {}
    _global: float = 1.0

    def fit(self, train_df, cfg=None):
        self._counts = train_df["end_town"].astype(str).value_counts().to_dict()
        self._global = float(np.median(list(self._counts.values())))

    def transform(self, df):
        g = self._global
        vals = [self._counts.get(str(t), g) for t in df["end_town"].tolist()]
        return pd.DataFrame({"dest_demand_score": vals}, index=df.index)


class OriginDemandBlock(FeatureBlock):
    name = "origin_demand"
    output_cols = ["origin_demand_score"]
    cost = "cheap"

    _counts: Dict[str, int] = {}
    _global: float = 1.0

    def fit(self, train_df, cfg=None):
        self._counts = train_df["start_town"].astype(str).value_counts().to_dict()
        self._global = float(np.median(list(self._counts.values())))

    def transform(self, df):
        g = self._global
        vals = [self._counts.get(str(t), g) for t in df["start_town"].tolist()]
        return pd.DataFrame({"origin_demand_score": vals}, index=df.index)


class OdPairStatsBlock(FeatureBlock):
    name = "od_pair_stats"
    output_cols = ["od_count", "od_mean_error"]
    cost = "expensive"

    _stats: Dict[tuple, Dict] = {}
    _global: Dict[str, float] = {}

    def fit(self, train_df, cfg=None):
        g_mean = float(train_df["eta_error"].mean())
        m = 50
        grp = train_df.groupby(["start_town", "end_town"])["eta_error"].agg(
            n="count", mean="mean"
        ).reset_index()
        grp["od_mean_error"] = (grp["n"] * grp["mean"] + m * g_mean) / (grp["n"] + m)
        self._stats = {
            (r["start_town"], r["end_town"]): {
                "od_count": int(r["n"]),
                "od_mean_error": float(r["od_mean_error"]),
            }
            for _, r in grp.iterrows()
        }
        g_count = float(grp["n"].median())
        self._global = {"od_count": g_count, "od_mean_error": g_mean}

    def transform(self, df):
        g = self._global
        rows = []
        for st, et in zip(df["start_town"].tolist(), df["end_town"].tolist()):
            rows.append(self._stats.get((st, et), g))
        return pd.DataFrame(rows, index=df.index)


class DemandByHourBlock(FeatureBlock):
    name = "demand_by_hour"
    output_cols = ["dest_demand_hour"]
    cost = "cheap"

    _counts: Dict[tuple, int] = {}
    _global: float = 1.0

    def fit(self, train_df, cfg=None):
        grp = train_df.groupby(["start_town", "hour"]).size().to_dict()
        self._counts = {k: v for k, v in grp.items()}
        self._global = float(np.median(list(self._counts.values())))

    def transform(self, df):
        g = self._global
        vals = [
            self._counts.get((st, h), g)
            for st, h in zip(df["start_town"].tolist(), df["hour"].tolist())
        ]
        return pd.DataFrame({"dest_demand_hour": vals}, index=df.index)


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: Dict[str, FeatureBlock] = {
    cls.name: cls()
    for cls in [
        BaseEtaBlock, TimeBasicBlock, TimeCyclicBlock, TimeFlagsBlock,
        GeoRawBlock, GeoDistanceBlock, GeoCenterBlock, GeoFlagsBlock,
        RegionCatBlock, RegionFreqBlock, RegionTargetEncBlock,
        DriverBiasBlock, DriverExperienceBlock,
        DestDemandBlock, OriginDemandBlock, OdPairStatsBlock,
        EtaDerivedBlock, DemandByHourBlock,
    ]
}


def _topo_sort(block_names: List[str]) -> List[str]:
    """Topological sort respecting depends_on."""
    order, visited, active = [], set(), set()

    def visit(name):
        if name in active:
            raise ValueError(f"Circular dependency detected at block '{name}'")
        if name in visited:
            return
        active.add(name)
        blk = REGISTRY[name]
        for dep in blk.depends_on:
            if dep in REGISTRY and dep in block_names:
                visit(dep)
        active.discard(name)
        visited.add(name)
        order.append(name)

    for name in block_names:
        visit(name)
    return order


def build_features(
    block_names: List[str],
    splits: Dict[str, "pd.DataFrame"],
    cfg=None,
) -> Dict[str, FeatureBundle]:
    """Build feature superset once; returns FeatureBundle per split."""
    # Ensure all required blocks are present for depends_on
    needed = set(block_names)
    for name in list(needed):
        for dep in REGISTRY[name].depends_on:
            if dep in REGISTRY:
                needed.add(dep)
    sorted_blocks = _topo_sort(list(needed))

    train_df = splits["train"]

    # Fit stateful blocks on train only
    for name in sorted_blocks:
        blk = REGISTRY[name]
        blk.fit(train_df, cfg)

    # Accumulate intermediate columns needed by EtaDerivedBlock
    all_cat_cols: List[str] = []
    superset: Dict[str, Dict[str, "pd.DataFrame"]] = {sp: {} for sp in splits}

    for name in sorted_blocks:
        blk = REGISTRY[name]
        for sp_name, sp_df in splits.items():
            # Augment sp_df with already-computed columns so depends_on blocks can read them
            augmented = sp_df.copy()
            for prev_name, prev_feats in superset[sp_name].items():
                for col in prev_feats.columns:
                    if col not in augmented.columns:
                        augmented[col] = prev_feats[col].values
            feats = blk.transform(augmented)
            superset[sp_name][name] = feats

        for col in blk.categorical_cols:
            if col not in all_cat_cols:
                all_cat_cols.append(col)

    bundles: Dict[str, FeatureBundle] = {}
    for sp_name, sp_df in splits.items():
        pieces = list(superset[sp_name].values())
        X = pd.concat(pieces, axis=1)

        # Leakage guard
        bad = set(X.columns) & FORBIDDEN_FEATURE_COLS
        if bad:
            raise RuntimeError(f"Leakage: forbidden columns in features: {bad}")

        y = sp_df["eta_error"].values
        bundles[sp_name] = FeatureBundle(X=X, y=y, cat_cols=all_cat_cols)

    return bundles
