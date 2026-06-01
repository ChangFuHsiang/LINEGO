import numpy as np
import polars as pl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, H3_RES_FINE, H3_RES_COARSE, TE_SMOOTH, SPLIT, SPECIAL_DAYS_2026, get_run_paths, new_run_dir

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
    print("WARNING: h3 not installed, skipping H3 features")


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


def add_time(df: pl.DataFrame) -> pl.DataFrame:
    h = pl.col("local_dt").dt.hour()
    special_dates = pl.Series("_sp", list(SPECIAL_DAYS_2026))

    df = df.with_columns(
        (2 * np.pi * h / 24).sin().alias("hour_sin"),
        (2 * np.pi * h / 24).cos().alias("hour_cos"),
        pl.col("local_dt").dt.weekday().alias("dow"),  # 1=Mon ... 7=Sun
        (pl.col("local_dt").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("driver_eta").log().alias("log_eta"),
        pl.col("local_dt").dt.hour().alias("hour"),
        pl.col("local_dt").dt.month().alias("month"),
    )

    # is_special_day
    date_str = df["date"].cast(pl.Utf8)
    is_special = date_str.is_in(list(SPECIAL_DAYS_2026)).cast(pl.Int8)
    df = df.with_columns(is_special.alias("is_special_day"))

    return df


def smoothed_te(train: pl.DataFrame, key: str, target: str, m: int):
    g = train[target].mean()
    agg = (
        train.group_by(key)
        .agg(pl.col(target).mean().alias("m_val"), pl.col(target).count().alias("n"))
        .with_columns(
            ((pl.col("n") * pl.col("m_val") + m * g) / (pl.col("n") + m)).alias(f"{key}_te")
        )
        .select([key, f"{key}_te"])
    )
    return agg, g


def oof_target_encoding(train: pl.DataFrame, key: str, target: str, m: int, n_folds: int = 5):
    """5-fold OOF target encoding for train set — prevents label leakage."""
    n = len(train)
    fold_ids = np.arange(n) % n_folds
    train = train.with_columns(pl.Series("_fold", fold_ids))
    te_vals = np.full(n, np.nan)
    g = train[target].mean()

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

    return train.with_columns(pl.Series(f"{key}_te", te_vals)).drop("_fold")


def main(run_dir: Path = None, target_col: str = "target_logratio"):
    if run_dir is None:
        run_dir = new_run_dir()
    paths = get_run_paths(run_dir)

    print("=== features.py ===")
    df = pl.read_parquet(paths["data_clean"])
    print(f"Loaded {df.height:,} rows")

    df = add_h3(df)
    df = add_time(df)

    # 時間連續切分
    is_test  = pl.col("date").cast(pl.Utf8) >= SPLIT["test_start"]
    is_valid = (pl.col("date").cast(pl.Utf8) >= SPLIT["valid_start"]) & ~is_test
    df = df.with_columns(
        pl.when(is_test).then(pl.lit("test"))
          .when(is_valid).then(pl.lit("valid"))
          .otherwise(pl.lit("train")).alias("split")
    )

    counts = df.group_by("split").len().sort("split")
    print(f"\nSplit counts:\n{counts}")

    train = df.filter(pl.col("split") == "train")

    # ==== Target Encoding ====
    # valid/test → 用全 train 算(無洩漏)
    # train     → 5-fold OOF
    for key, m in TE_SMOOTH.items():
        if key not in df.columns:
            print(f"  Skipping TE for {key} (column missing)")
            continue
        print(f"  TE: {key} (m={m})...")
        agg, g = smoothed_te(train, key, target_col, m)

        # OOF for train
        train_oof = oof_target_encoding(train, key, target_col, m)
        train_te_map = dict(zip(
            train_oof["trip_id"].to_list(),
            train_oof[f"{key}_te"].to_list()
        ))

        # 全域 map for valid/test
        global_map = dict(zip(agg[key].to_list(), agg[f"{key}_te"].to_list()))

        def apply_te(df_part: pl.DataFrame, split: str) -> pl.Series:
            keys_list = df_part[key].to_list()
            if split == "train":
                ids = df_part["trip_id"].to_list()
                vals = [train_te_map.get(tid, g) for tid in ids]
            else:
                vals = [global_map.get(k, g) for k in keys_list]
            return pl.Series(f"{key}_te", vals)

        parts = []
        for sp in ["train", "valid", "test"]:
            part = df.filter(pl.col("split") == sp)
            te_col = apply_te(part, sp)
            part = part.with_columns(te_col)
            parts.append(part)
        df = pl.concat(parts)

    # 叫車量特徵:同 town × hour 的歷史平均單量(只用 train 算)
    print("  Computing demand proxy (town × hour)...")
    demand = (
        train.group_by(["start_town", "hour"])
        .agg(pl.len().alias("demand_town_hour"))
    )
    df = df.join(demand, on=["start_town", "hour"], how="left")
    df = df.with_columns(pl.col("demand_town_hour").fill_null(1))

    df.write_parquet(paths["data_feats"])
    print(f"\nFeatures written: {paths['data_feats']}")
    print(f"Columns: {[c for c in df.columns if '_te' in c or c in ['hour_sin','hour_cos','dow','is_weekend','log_eta','is_special_day','demand_town_hour']]}")


if __name__ == "__main__":
    main()
