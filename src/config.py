from pathlib import Path
from datetime import datetime

DATA_RAW = Path("data/raw/trip_stats_eta.parquet")


def get_run_paths(run_dir: Path) -> dict:
    """Return all per-run paths derived from run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return {
        "run_dir":    run_dir,
        "data_clean": run_dir / "trips_clean.parquet",
        "data_feats": run_dir / "feats.parquet",
        "models_dir": run_dir / "models",
        "metrics_dir":run_dir / "metrics",
        "figures_dir":run_dir / "figures",
    }


def new_run_dir() -> Path:
    """Create a timestamped run directory under outputs/."""
    return Path("outputs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

TZ_OFFSET_HOURS = 8  # request_time 是 UTC,+8 轉台灣

H3_RES_FINE   = 9   # ~170m
H3_RES_COARSE = 8   # 兜底

SPLIT = {
    "valid_start": "2026-04-23",
    "test_start":  "2026-05-07",
}

TE_SMOOTH = {"start_h3": 20, "start_town": 50, "did_hash": 30, "uid_hash": 200}

LGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
)

# 台灣 2026 特殊日(春節、連假、颱風等)
SPECIAL_DAYS_2026 = {
    # 農曆春節(2/14 補班~2/22 元宵)
    "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17",
    "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22",
    # 228 紀念日連假
    "2026-02-27", "2026-02-28", "2026-03-01",
    # 清明連假
    "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06",
    # 端午節
    "2026-06-19", "2026-06-20", "2026-06-21",
    # 跨年
    "2026-01-01",
}
