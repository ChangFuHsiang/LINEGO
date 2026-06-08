"""Load and clean raw data — identical protocol to src/."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, DATA_CLEAN, TZ_OFFSET_HOURS


def main():
    print("=== mix/load_clean.py ===")
    print(f"Reading: {DATA_RAW}")

    lf = pl.scan_parquet(DATA_RAW)
    df = lf.collect()
    print(f"Raw rows: {df.height:,}")

    # Cleaning redlines — same as src/
    lf = pl.scan_parquet(DATA_RAW).filter(
        (pl.col("driver_eta") > 0)
        & (pl.col("time_accept_to_arrive") > 0)
        & pl.col("start_county").is_not_null()
        & pl.col("start_town").is_not_null()
    )

    lf = lf.with_columns(
        (pl.from_epoch("request_time", time_unit="s")
           .dt.offset_by(f"{TZ_OFFSET_HOURS}h")).alias("local_dt")
    )

    df = lf.collect()
    print(f"After basic cleaning: {df.height:,} rows")

    # P99.9 upper cap — same as src/
    hi_arrive = df["time_accept_to_arrive"].quantile(0.999)
    hi_eta    = df["driver_eta"].quantile(0.999)
    print(f"P99.9 time_accept_to_arrive: {hi_arrive:.0f}s  driver_eta: {hi_eta:.0f}s")

    df = df.filter(
        (pl.col("time_accept_to_arrive") < hi_arrive)
        & (pl.col("driver_eta") < hi_eta)
    )
    print(f"After P99.9 cap: {df.height:,} rows")

    # Log-ratio target (same as src/)
    df = df.with_columns(
        (pl.col("time_accept_to_arrive") / pl.col("driver_eta")).log().alias("target_logratio"),
        (pl.col("time_accept_to_arrive") - pl.col("driver_eta")).alias("target_additive"),
        pl.col("local_dt").dt.date().cast(pl.Utf8).alias("date"),
    )

    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    lr = df["target_logratio"]
    print(f"Log-ratio: mean={lr.mean():.3f} std={lr.std():.3f} "
          f"p50={lr.quantile(0.5):.3f} p90={lr.quantile(0.9):.3f}")

    df.write_parquet(DATA_CLEAN)
    print(f"Saved: {DATA_CLEAN}  ({df.height:,} rows)")


if __name__ == "__main__":
    main()
