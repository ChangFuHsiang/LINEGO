"""Load, clean, split, and audit the raw ETA dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl

from eta_pipeline.config import RunConfig


LEAKAGE_COLS = {"time_accept_to_arrive", "time_start_to_finish", "trip_id", "end_address"}
REQUIRED_COLS = {
    "trip_id", "did_hash", "uid_hash",
    "start_lat", "start_lng", "start_county", "start_town",
    "end_lat", "end_lng", "end_county", "end_town",
    "request_time", "driver_eta", "time_accept_to_arrive",
}


def load_raw(path: str | Path) -> pl.DataFrame:
    df = pl.scan_parquet(path).collect()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def clean(df: pl.DataFrame, cfg: RunConfig) -> pl.DataFrame:
    tz = cfg.split.tz_offset_hours

    # Parse timestamp → local datetime
    df = df.with_columns(
        (pl.from_epoch("request_time", time_unit="s")
           .dt.offset_by(f"{tz}h")).alias("local_dt")
    )

    # Null-fill end_* with start_* so geo blocks don't break
    df = df.with_columns([
        pl.col("end_lat").fill_null(pl.col("start_lat")),
        pl.col("end_lng").fill_null(pl.col("start_lng")),
        pl.col("end_county").fill_null(pl.col("start_county")),
        pl.col("end_town").fill_null(pl.col("start_town")),
    ])
    df = df.with_columns([
        pl.col("start_county").fill_null(pl.lit("unknown")),
        pl.col("start_town").fill_null(pl.lit("unknown")),
        pl.col("end_county").fill_null(pl.lit("unknown")),
        pl.col("end_town").fill_null(pl.lit("unknown")),
    ])

    # Drop rows with invalid core values
    df = df.filter(
        (pl.col("driver_eta") > 0)
        & (pl.col("time_accept_to_arrive") > 0)
    )

    # Derived time columns
    df = df.with_columns([
        pl.col("local_dt").dt.month().alias("month"),
        pl.col("local_dt").dt.year().alias("year"),
        pl.col("local_dt").dt.hour().alias("hour"),
        pl.col("local_dt").dt.weekday().alias("day_of_week"),  # 1=Mon … 7=Sun
        pl.col("local_dt").dt.day().alias("day_of_month"),
        pl.col("local_dt").dt.date().cast(pl.Utf8).alias("date_str"),
    ])

    # LNY flag
    lny_start, lny_end = cfg.split.lny_window
    df = df.with_columns(
        ((pl.col("date_str") >= lny_start) & (pl.col("date_str") <= lny_end))
        .cast(pl.Int8).alias("is_lunar_new_year")
    )

    return df


def add_target(df: pl.DataFrame, cfg: RunConfig) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("time_accept_to_arrive") - pl.col("driver_eta")).alias("eta_error")
    )

    if cfg.data.clip_target_quantiles is not None:
        lo_q, hi_q = cfg.data.clip_target_quantiles
        lo = df["eta_error"].quantile(lo_q)
        hi = df["eta_error"].quantile(hi_q)
        df = df.filter(
            (pl.col("eta_error") >= lo) & (pl.col("eta_error") <= hi)
        )

    return df


def make_splits(df: pl.DataFrame, cfg: RunConfig) -> Dict[str, pd.DataFrame]:
    train_mask = pl.col("month").is_in(cfg.split.train_months)
    val_mask   = pl.col("month").is_in(cfg.split.val_months)
    test_mask  = pl.col("month").is_in(cfg.split.test_months)

    # Cast all non-primitive types to avoid pyarrow dependency in to_pandas()
    for col_name, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.Datetime or dtype == pl.Date or str(dtype).startswith("Datetime"):
            df = df.with_columns(
                pl.col(col_name).dt.to_string("%Y-%m-%d %H:%M:%S")
            )
        elif str(dtype).startswith("Duration"):
            df = df.with_columns(
                pl.col(col_name).cast(pl.Int64)
            )

    def _to_pandas(lf: pl.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(lf.to_dict(as_series=False))

    splits = {
        "train": _to_pandas(df.filter(train_mask)),
        "val":   _to_pandas(df.filter(val_mask)),
        "test":  _to_pandas(df.filter(test_mask)),
    }

    for name, sdf in splits.items():
        if sdf.empty:
            raise ValueError(f"Split '{name}' is empty. Check split.{name}_months in config.")
        date_range = f"{sdf['date_str'].min()} ~ {sdf['date_str'].max()}"
        print(f"  {name:5s}: {len(sdf):>8,} rows  [{date_range}]")

    return splits


def audit(splits: Dict[str, pd.DataFrame], cfg: RunConfig) -> dict:
    train = splits["train"]
    report: dict = {}

    report["rows"] = {k: len(v) for k, v in splits.items()}

    null_pct = {}
    for col in train.columns:
        n = train[col].isna().sum()
        if n > 0:
            null_pct[col] = round(n / len(train) * 100, 2)
    report["null_pct_train"] = null_pct

    eta = train["driver_eta"]
    err = train["eta_error"]
    report["driver_eta_stats"] = {
        "mean": float(eta.mean()), "std": float(eta.std()),
        "p50": float(np.percentile(eta, 50)), "p90": float(np.percentile(eta, 90)),
        "p99": float(np.percentile(eta, 99)),
    }
    report["eta_error_stats"] = {
        "mean": float(err.mean()), "std": float(err.std()),
        "p10": float(np.percentile(err, 10)),
        "p50": float(np.percentile(err, 50)),
        "p90": float(np.percentile(err, 90)),
        "p99": float(np.percentile(err, 99)),
    }

    lny_rows = train["is_lunar_new_year"].sum()
    report["lny_rows_train"] = int(lny_rows)
    report["lny_window"] = cfg.split.lny_window

    end_null = train["end_lat"].isna().sum()
    report["end_lat_nulls_train"] = int(end_null)

    clip_q = cfg.data.clip_target_quantiles
    report["clip_rule"] = f"quantiles {clip_q}" if clip_q else "none"

    return report
