"""mix/ pipeline constants — common protocol + mix-specific settings."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_RAW   = Path("data/raw/trip_stats_eta.parquet")
MIX_DIR    = Path("outputs/run_mix")
DATA_CLEAN = MIX_DIR / "trips_clean.parquet"
DATA_FEATS = MIX_DIR / "feats.parquet"

for d in [MIX_DIR, MIX_DIR / "models", MIX_DIR / "metrics", MIX_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

MODELS_DIR  = MIX_DIR / "models"
METRICS_DIR = MIX_DIR / "metrics"
FIGURES_DIR = MIX_DIR / "figures"

# ── Common protocol (must match src/) ─────────────────────────────────────────
TZ_OFFSET_HOURS = 8

# Continuous time splits: train < valid < test
SPLIT = {
    "valid_start": "2026-04-23",
    "test_start":  "2026-05-07",
}

# ── LNY exclusion for driver stats (inclusive) ────────────────────────────────
# 2026 Lunar New Year period — exclude from driver bias stats to avoid
# regime contamination; same window as CLAUDE.md §8.1
LNY_EXCLUDE_START = "2026-02-14"
LNY_EXCLUDE_END   = "2026-02-22"

# ── H3 resolution ─────────────────────────────────────────────────────────────
H3_RES_FINE   = 9   # ~170m
H3_RES_COARSE = 8

# ── Special days (for is_special_day feature, kept from src/) ─────────────────
SPECIAL_DAYS_2026 = {
    "2026-02-14","2026-02-15","2026-02-16","2026-02-17",
    "2026-02-18","2026-02-19","2026-02-20","2026-02-21","2026-02-22",
    "2026-02-27","2026-02-28","2026-03-01",
    "2026-04-03","2026-04-04","2026-04-05","2026-04-06",
    "2026-06-19","2026-06-20","2026-06-21",
    "2026-01-01",
}

# ── Target encoding smoothing ─────────────────────────────────────────────────
TE_SMOOTH_H3    = 20   # fewer trips per H3 cell → stronger smoothing
TE_SMOOTH_TOWN  = 50
TE_FOLDS        = 5

# ── Driver stats smoothing ────────────────────────────────────────────────────
DRIVER_SMOOTH_M      = 30   # prior weight toward global mean/median
DRIVER_MIN_TRIPS_STD = 10   # fill std with global median if n < 10

# ── Optuna ────────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS    = 80
OPTUNA_BOOST_ROUND = 500   # tuning only (fast); final uses best_iter * ROUND_INFLATION
OPTUNA_EARLY_STOP  = 30
ROUND_INFLATION    = 1.2   # final_rounds = best_iteration * 1.2
OPTUNA_DB          = "outputs/run_mix/optuna.db"

SEED = 42

# ── Feature list ──────────────────────────────────────────────────────────────
# log_eta + time (6) + driver stats (3) + rider activity (2)
# + h3/town TE (2) + demand (1) + end_town categorical (1) = 16 features
FEATURES_MIX = [
    # API signal
    "log_eta",
    # Time (cyclic + calendar + flags)
    "hour_sin", "hour_cos", "dow", "is_weekend", "is_special_day",
    # Driver — 3 stats replacing single did_hash_te
    "driver_avg_logratio", "driver_median_logratio", "driver_std_logratio",
    # Rider — activity replacing uid_hash_te
    "uid_trip_count_train", "uid_days_since_first",
    # Location TE (kept from src/)
    "start_h3_te", "start_town_te",
    # Demand (kept from src/)
    "demand_town_hour",
    # Destination categorical (new — native LGBM categorical, no TE)
    "end_town",
]

# end_town must be passed as categorical_feature to LightGBM
CAT_FEATURES = ["end_town"]

# ── LightGBM default params (used when Optuna is skipped / for warmup) ────────
LGB_DEFAULT = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
)
