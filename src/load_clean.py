import polars as pl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, DATA_CLEAN, TZ_OFFSET_HOURS


def main():
    print("=== load_clean.py ===")
    print(f"Reading: {DATA_RAW}")

    lf = pl.scan_parquet(DATA_RAW)

    # 印原始統計
    df_raw = lf.collect()
    print(f"Raw rows: {df_raw.height:,}")
    print(f"Columns: {df_raw.columns}")
    print(f"\ndriver_eta stats:\n{df_raw['driver_eta'].describe()}")
    print(f"\ntime_accept_to_arrive stats:\n{df_raw['time_accept_to_arrive'].describe()}")

    # 清洗紅線
    lf = lf.filter(
        (pl.col("driver_eta") > 0)
        & (pl.col("time_accept_to_arrive") > 0)
        & pl.col("start_county").is_not_null()
        & pl.col("start_town").is_not_null()
    )

    # 時間:Unix 秒 → 台灣當地 datetime
    lf = lf.with_columns(
        (pl.from_epoch("request_time", time_unit="s")
           .dt.offset_by(f"{TZ_OFFSET_HOURS}h")).alias("local_dt")
    )

    df = lf.collect()
    print(f"\nAfter basic cleaning: {df.height:,} rows")

    # P99.9 上限
    hi_arrive = df["time_accept_to_arrive"].quantile(0.999)
    hi_eta    = df["driver_eta"].quantile(0.999)
    print(f"\nP99.9 time_accept_to_arrive: {hi_arrive:.0f}s ({hi_arrive/60:.1f} min)")
    print(f"P99.9 driver_eta:            {hi_eta:.0f}s ({hi_eta/60:.1f} min)")

    df = df.filter(
        (pl.col("time_accept_to_arrive") < hi_arrive)
        & (pl.col("driver_eta") < hi_eta)
    )
    print(f"After P99.9 cap: {df.height:,} rows")

    # targets
    df = df.with_columns(
        (pl.col("time_accept_to_arrive") / pl.col("driver_eta")).log().alias("target_logratio"),
        (pl.col("time_accept_to_arrive") - pl.col("driver_eta")).alias("target_additive"),
        pl.col("local_dt").dt.date().alias("date"),
    )

    # 印殘差分布確認系統性低估
    residual = df["target_additive"]
    print(f"\n=== 殘差分布(time_accept_to_arrive - driver_eta) ===")
    print(f"Mean:   {residual.mean():.1f}s")
    print(f"Median: {residual.median():.1f}s")
    print(f"Std:    {residual.std():.1f}s")
    print(f"MAE:    {(df['time_accept_to_arrive'] - df['driver_eta']).abs().mean():.1f}s")
    import numpy as np
    err = (df["time_accept_to_arrive"] - df["driver_eta"]).to_numpy()
    rmse = np.sqrt((err**2).mean())
    print(f"RMSE:   {rmse:.1f}s")
    print(f"Late>60s:  {(residual > 60).mean()*100:.1f}%")
    print(f"Early>60s: {(residual < -60).mean()*100:.1f}%")

    lr = df["target_logratio"]
    print(f"\n=== Log-ratio 分布 ===")
    print(f"Mean: {lr.mean():.3f}, Std: {lr.std():.3f}")
    print(f"P50: {lr.quantile(0.5):.3f}, P90: {lr.quantile(0.9):.3f}, P99: {lr.quantile(0.99):.3f}")

    print(f"\nDate range: {df['date'].min()} ~ {df['date'].max()}")

    df.write_parquet(DATA_CLEAN)
    print(f"\nSaved: {DATA_CLEAN}  ({df.height:,} rows)")


if __name__ == "__main__":
    main()
