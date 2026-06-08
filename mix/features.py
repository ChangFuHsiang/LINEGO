"""Feature engineering — mix pipeline with 5 modifications over src/."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_CLEAN, DATA_FEATS, SPLIT, SPECIAL_DAYS_2026,
    H3_RES_FINE, H3_RES_COARSE, TE_SMOOTH_H3, TE_SMOOTH_TOWN, TE_FOLDS,
    LNY_EXCLUDE_START, LNY_EXCLUDE_END,
    DRIVER_SMOOTH_M, DRIVER_MIN_TRIPS_STD, SEED,
)

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
    print("WARNING: h3 not installed — start_h3 features will be 'unknown'")


# ── H3 ────────────────────────────────────────────────────────────────────────

def add_h3(df: pl.DataFrame) -> pl.DataFrame:
    if not HAS_H3:
        return df.with_columns(
            pl.lit("unknown").alias("start_h3"),
            pl.lit("unknown").alias("start_h3c"),
        )
    lat = df["start_lat"].to_numpy()
    lng = df["start_lng"].to_numpy()
    fine   = [h3.latlng_to_cell(float(a), float(b), H3_RES_FINE)   for a, b in zip(lat, lng)]
    coarse = [h3.latlng_to_cell(float(a), float(b), H3_RES_COARSE) for a, b in zip(lat, lng)]
    return df.with_columns(
        pl.Series("start_h3",  fine),
        pl.Series("start_h3c", coarse),
    )


# ── Time features (same as src/) ──────────────────────────────────────────────

def add_time(df: pl.DataFrame) -> pl.DataFrame:
    h = pl.col("local_dt").dt.hour()
    df = df.with_columns(
        (2 * np.pi * h / 24).sin().alias("hour_sin"),
        (2 * np.pi * h / 24).cos().alias("hour_cos"),
        pl.col("local_dt").dt.weekday().alias("dow"),
        (pl.col("local_dt").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("driver_eta").log().alias("log_eta"),
        pl.col("local_dt").dt.hour().alias("hour"),
        pl.col("local_dt").dt.month().alias("month"),
    )
    date_str = df["date"].cast(pl.Utf8)
    is_special = date_str.is_in(list(SPECIAL_DAYS_2026)).cast(pl.Int8)
    return df.with_columns(is_special.alias("is_special_day"))


# ── Target encoding helpers ────────────────────────────────────────────────────

def smoothed_te(train: pl.DataFrame, key: str, target: str, m: int):
    g = float(train[target].mean())
    agg = (
        train.group_by(key)
        .agg(pl.col(target).mean().alias("m_val"), pl.col(target).count().alias("n"))
        .with_columns(
            ((pl.col("n") * pl.col("m_val") + m * g) / (pl.col("n") + m)).alias(f"{key}_te")
        )
        .select([key, f"{key}_te"])
    )
    return agg, g


def oof_target_encoding(train: pl.DataFrame, key: str, target: str,
                        m: int, n_folds: int = TE_FOLDS):
    """5-fold OOF TE — prevents train-set leakage."""
    n = len(train)
    fold_ids = np.arange(n) % n_folds
    train = train.with_columns(pl.Series("_fold", fold_ids))
    te_vals = np.full(n, np.nan)
    g = float(train[target].mean())

    for fold in range(n_folds):
        other = train.filter(pl.col("_fold") != fold)
        agg = (
            other.group_by(key)
            .agg(pl.col(target).mean().alias("m_val"), pl.col(target).count().alias("n"))
            .with_columns(
                ((pl.col("n") * pl.col("m_val") + m * g) / (pl.col("n") + m)).alias("te")
            )
            .select([key, "te"])
        )
        lookup = dict(zip(agg[key].to_list(), agg["te"].to_list()))
        fold_mask = fold_ids == fold
        keys_fold = train.filter(pl.col("_fold") == fold)[key].to_list()
        te_vals[fold_mask] = [lookup.get(k, g) for k in keys_fold]

    return train.with_columns(pl.Series(f"{key}_te", te_vals)).drop("_fold"), g


# ── Modification 1: Driver 3-stat features ────────────────────────────────────
# Replaces did_hash_te with avg/median/std of log-ratio, computed train-only
# LNY period excluded to avoid seasonal regime contamination.

def _driver_stats_from(df_no_lny: pd.DataFrame, g_mean: float,
                        g_median: float, g_std: float) -> dict:
    """Compute smoothed driver stats from a given dataset (LNY already excluded)."""
    m = DRIVER_SMOOTH_M
    grp = df_no_lny.groupby("did_hash")["target_logratio"].agg(
        ["mean", "median", "std", "count"]
    ).reset_index()
    grp.columns = ["did_hash", "mean", "median", "std", "n"]

    grp["driver_avg_logratio"] = (
        (grp["n"] * grp["mean"] + m * g_mean) / (grp["n"] + m)
    )
    grp["driver_median_logratio"] = (
        (grp["n"] * grp["median"] + m * g_median) / (grp["n"] + m)
    )
    # Std: fill with global_std when n < DRIVER_MIN_TRIPS_STD (unstable)
    grp["driver_std_logratio"] = np.where(
        grp["n"] >= DRIVER_MIN_TRIPS_STD, grp["std"].fillna(g_std), g_std
    )

    return grp.set_index("did_hash")[
        ["driver_avg_logratio", "driver_median_logratio", "driver_std_logratio"]
    ].to_dict("index")


def add_driver_stats(train_pd: pd.DataFrame,
                     splits_pd: dict) -> dict:
    """
    train_pd  : full train split as pandas
    splits_pd : {"train": ..., "valid": ..., "test": ...}
    Returns   : dict of split → pd.DataFrame with 3 driver stat columns
    """
    # Global stats (LNY excluded) — used as prior and for unseen drivers
    lny_mask = (
        (train_pd["date"] >= LNY_EXCLUDE_START)
        & (train_pd["date"] <= LNY_EXCLUDE_END)
    )
    train_no_lny = train_pd[~lny_mask]
    g_mean   = float(train_no_lny["target_logratio"].mean())
    g_median = float(train_no_lny["target_logratio"].median())
    g_std    = float(train_no_lny["target_logratio"].std())
    defaults = {"driver_avg_logratio": g_mean,
                "driver_median_logratio": g_median,
                "driver_std_logratio": g_std}

    # ── valid/test: use full-train stats (no OOF needed) ──────────────────────
    full_stats = _driver_stats_from(train_no_lny, g_mean, g_median, g_std)

    def apply_stats(df_split: pd.DataFrame, lookup: dict) -> pd.DataFrame:
        rows = [lookup.get(did, defaults) for did in df_split["did_hash"]]
        return pd.DataFrame(rows, index=df_split.index)

    result = {}
    for sp, df_sp in splits_pd.items():
        if sp == "train":
            continue  # handled by OOF below
        result[sp] = apply_stats(df_sp, full_stats)

    # ── train: 5-fold OOF ─────────────────────────────────────────────────────
    n = len(train_pd)
    # Deterministic fold assignment matching overall row order (same as src/ OOF)
    rng = np.random.default_rng(SEED)
    fold_ids = rng.permutation(n) % TE_FOLDS  # shuffled for balance

    cols = ["driver_avg_logratio", "driver_median_logratio", "driver_std_logratio"]
    oof_arr = np.full((n, 3), [g_mean, g_median, g_std])

    for fold in range(TE_FOLDS):
        other_mask = fold_ids != fold
        fold_mask  = fold_ids == fold

        other = train_pd.iloc[other_mask]
        other_no_lny = other[
            ~((other["date"] >= LNY_EXCLUDE_START)
              & (other["date"] <= LNY_EXCLUDE_END))
        ]
        fold_stats = _driver_stats_from(other_no_lny, g_mean, g_median, g_std)

        fold_dids = train_pd.iloc[fold_mask]["did_hash"].tolist()
        for i_local, did in zip(np.where(fold_mask)[0], fold_dids):
            if did in fold_stats:
                s = fold_stats[did]
                oof_arr[i_local] = [
                    s["driver_avg_logratio"],
                    s["driver_median_logratio"],
                    s["driver_std_logratio"],
                ]

    result["train"] = pd.DataFrame(
        oof_arr, columns=cols, index=train_pd.index
    )
    return result


# ── Modification 2: Rider activity features ───────────────────────────────────
# Replaces uid_hash_te. Both stats computed from train only and merged.
# New riders (not in train) get 0 for both.

def add_uid_activity(train_pd: pd.DataFrame,
                     splits_pd: dict) -> dict:
    uid_counts     = train_pd.groupby("uid_hash").size().to_dict()
    uid_first_date = train_pd.groupby("uid_hash")["date"].min().to_dict()

    result = {}
    for sp, df_sp in splits_pd.items():
        trip_count = df_sp["uid_hash"].map(uid_counts).fillna(0).astype(int)

        uid_first = df_sp["uid_hash"].map(uid_first_date)
        uid_days  = (
            pd.to_datetime(df_sp["date"]) - pd.to_datetime(uid_first)
        ).dt.days.fillna(0).clip(lower=0).astype(int)

        result[sp] = pd.DataFrame({
            "uid_trip_count_train": trip_count.values,
            "uid_days_since_first": uid_days.values,
        }, index=df_sp.index)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== mix/features.py ===")
    df_pl = pl.read_parquet(DATA_CLEAN)
    print(f"Loaded {df_pl.height:,} rows")

    df_pl = add_h3(df_pl)
    df_pl = add_time(df_pl)

    # Time-continuous split (same as src/)
    is_test  = pl.col("date") >= SPLIT["test_start"]
    is_valid = (pl.col("date") >= SPLIT["valid_start"]) & ~is_test
    df_pl = df_pl.with_columns(
        pl.when(is_test).then(pl.lit("test"))
          .when(is_valid).then(pl.lit("valid"))
          .otherwise(pl.lit("train")).alias("split")
    )

    counts = df_pl.group_by("split").len().sort("split")
    print(f"\nSplit counts:\n{counts}")

    train_pl = df_pl.filter(pl.col("split") == "train")
    target_col = "target_logratio"

    # ── src/-style TE for start_h3 and start_town (kept from src/) ────────────
    print("\nBuilding H3/town target encodings...")
    for key, m in [("start_h3", TE_SMOOTH_H3), ("start_town", TE_SMOOTH_TOWN)]:
        if key not in df_pl.columns:
            print(f"  Skipping {key} (column missing)")
            continue
        print(f"  OOF TE: {key} (m={m})")
        agg, g = smoothed_te(train_pl, key, target_col, m)

        train_oof, _ = oof_target_encoding(train_pl, key, target_col, m)
        train_te_map = dict(zip(
            train_oof["trip_id"].to_list(),
            train_oof[f"{key}_te"].to_list()
        ))
        global_map = dict(zip(agg[key].to_list(), agg[f"{key}_te"].to_list()))

        def apply_te(df_part: pl.DataFrame, sp: str) -> pl.Series:
            keys_list = df_part[key].to_list()
            if sp == "train":
                ids = df_part["trip_id"].to_list()
                vals = [train_te_map.get(tid, g) for tid in ids]
            else:
                vals = [global_map.get(k, g) for k in keys_list]
            return pl.Series(f"{key}_te", vals)

        parts = []
        for sp in ["train", "valid", "test"]:
            part = df_pl.filter(pl.col("split") == sp)
            part = part.with_columns(apply_te(part, sp))
            parts.append(part)
        df_pl = pl.concat(parts)

    # ── Demand proxy (kept from src/) ─────────────────────────────────────────
    print("  Computing demand_town_hour proxy...")
    train_refresh = df_pl.filter(pl.col("split") == "train")
    demand = (
        train_refresh.group_by(["start_town", "hour"])
        .agg(pl.len().alias("demand_town_hour"))
    )
    df_pl = df_pl.join(demand, on=["start_town", "hour"], how="left")
    df_pl = df_pl.with_columns(pl.col("demand_town_hour").fill_null(1))

    # ── Convert to pandas for the new mix-specific features ───────────────────
    splits_pd = {
        sp: df_pl.filter(pl.col("split") == sp).to_pandas()
        for sp in ["train", "valid", "test"]
    }
    train_pd = splits_pd["train"]

    # ── Modification 1: Driver 3-stat features ────────────────────────────────
    print("  Building driver 3-stat features (OOF, LNY excluded)...")
    driver_dfs = add_driver_stats(train_pd, splits_pd)

    # ── Modification 2: Rider activity features ───────────────────────────────
    print("  Building uid activity features...")
    uid_dfs = add_uid_activity(train_pd, splits_pd)

    # ── Merge everything back and re-build polars DataFrame ───────────────────
    parts = []
    for sp in ["train", "valid", "test"]:
        base_pl  = df_pl.filter(pl.col("split") == sp)
        base_pd  = base_pl.to_pandas().copy()

        for col, ser in driver_dfs[sp].items():
            base_pd[col] = ser.values
        for col, ser in uid_dfs[sp].items():
            base_pd[col] = ser.values

        # end_town: ensure it's a string (categorical for LGBM)
        base_pd["end_town"] = base_pd["end_town"].fillna("unknown").astype(str)

        parts.append(pl.from_pandas(base_pd))

    df_final = pl.concat(parts)
    df_final.write_parquet(DATA_FEATS)

    feat_cols = [c for c in df_final.columns
                 if c in [
                     "log_eta","hour_sin","hour_cos","dow","is_weekend","is_special_day",
                     "driver_avg_logratio","driver_median_logratio","driver_std_logratio",
                     "uid_trip_count_train","uid_days_since_first",
                     "start_h3_te","start_town_te","demand_town_hour","end_town",
                 ]]
    print(f"\nFeatures written: {DATA_FEATS}")
    print(f"Feature columns: {feat_cols}")

    # Sanity check: driver stats should differ between train folds
    tr = df_final.filter(pl.col("split") == "train")["driver_avg_logratio"]
    va = df_final.filter(pl.col("split") == "valid")["driver_avg_logratio"]
    print(f"\nSanity — driver_avg_logratio train mean={tr.mean():.4f} valid mean={va.mean():.4f}")


if __name__ == "__main__":
    main()
