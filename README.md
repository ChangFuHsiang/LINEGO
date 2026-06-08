# LINEGO — ETA Correction Pipeline

A machine learning pipeline that corrects systematic underestimation in a ride-hailing platform's driver arrival time (ETA) predictions.

---

## The Problem

The platform's routing API provides `driver_eta` — how many seconds until the driver reaches the passenger pickup point. This estimate is consistently **too optimistic**: drivers arrive ~76 seconds later than promised on average. This causes two problems:

1. Dispatch may select the wrong driver
2. Users see "3 minutes" but wait 6 minutes

---

## The Approach: Residual Learning

Instead of rebuilding a routing engine from scratch, each model learns **how wrong the API is**, then corrects it:

```
# Log-ratio target (Pipeline A / B)
corrected ETA = driver_eta × exp(model_prediction)

# Additive residual target (Pipeline C)
corrected ETA = driver_eta + model_prediction
```

This mirrors industry approaches used by Uber (DeepETA) and DiDi (WDR).

---

## Dataset

- **File:** `data/raw/trip_stats_eta.parquet`
- **Size:** 1.55M trips, Jan–May 2026, sourced from LINE GO
- **Key columns:** `driver_eta` (API estimate), `time_accept_to_arrive` (ground truth), pickup coordinates, timestamps, driver/passenger IDs

Download the dataset:

```bash
conda activate linego
python -m gdown "1hfOxA8nA1TZV1SA5GFblbhMMaDS7kXy4" -O data/raw/trip_stats_eta.parquet
```

---

## Setup

```bash
conda env create -f environment.yaml
conda activate linego
pip install h3   # required for H3 spatial features
```

---

## Project Structure

```
LINEGO/
├── environment.yaml
├── Draft.md                        # 四條 pipeline 的完整方法說明
├── data/
│   └── raw/trip_stats_eta.parquet  # 原始資料（勿修改）
│
├── src/                            # Pipeline A — src 純淨版
│   ├── config.py
│   ├── load_clean.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── mix/                            # Pipeline B — mix 強化版
│   ├── config.py
│   ├── load_clean.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run.py
│   └── compare.py
│
├── final/                          # Pipeline C — final 自動搜尋版
│   ├── config/
│   │   ├── default.yaml
│   │   └── quick.yaml
│   ├── src/eta_pipeline/           # 可安裝套件（18 個 FeatureBlock）
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── features.py
│   │   ├── feature_search.py
│   │   ├── tuning.py
│   │   ├── model.py
│   │   ├── metrics.py
│   │   ├── report.py
│   │   └── run.py
│   ├── tests/
│   └── pyproject.toml
│
├── run.py                          # Pipeline A 的一鍵執行入口
│
├── outputs/
│   ├── run_src/                    # Pipeline A 的輸出
│   │   ├── metrics/all_metrics.json
│   │   └── figures/
│   ├── run_mix/                    # Pipeline B 的輸出
│   │   ├── metrics/
│   │   └── figures/
│   └── figures_comparison/        # A vs B 對照圖
│
├── artifacts/                      # Pipeline C 的 Optuna 結果與 feature 選擇
│   ├── selected_*.json
│   └── best_params_*.json
│
└── reports/                        # Pipeline C 的文字報告
```

---

## Three Pipelines (LightGBM-based)

### Pipeline A — `src/` 純淨版

The baseline LightGBM pipeline with manually selected features and log-ratio target.

```bash
python run.py --run-dir outputs/run_src
```

**Features (11):** `log_eta`, `hour_sin/cos`, `dow`, `is_weekend`, `is_special_day`, `start_h3_te`, `start_town_te`, `did_hash_te`, `uid_hash_te`, `demand_town_hour`

**Results (test: May 7–20):**

| Model | MAE | RMSE | Within 60s | P90 | MAE Improvement |
|---|---|---|---|---|---|
| Baseline (driver_eta) | 88.1s | 128.0s | 49.6% | 196s | — |
| LightGBM (L2) | 76.6s | 111.1s | 54.2% | 167s | -13.2% |
| Quantile q50 | 76.1s | 110.2s | 54.2% | 165s | -13.6% |

Quantile [q10, q90] coverage: **79.7%** (target ~80%)

---

### Pipeline B — `mix/` 強化版

Enhanced features: driver statistics (avg/median/std instead of single TE), rider activity features, and `end_town` as a native categorical.

```bash
python mix/run.py
```

**Key changes from A:**
- `did_hash_te` → `driver_avg/median/std_logratio` (LNY-excluded, 5-fold OOF)
- `uid_hash_te` → `uid_trip_count_train` + `uid_days_since_first`
- New: `end_town` (native LightGBM categorical)
- New: Permutation importance (5 repeats on validation set)

**Results (test: May 7–20):**

| Model | MAE | RMSE | Within 60s | P90 | MAE Improvement |
|---|---|---|---|---|---|
| Baseline | 88.1s | 128.0s | 49.6% | 196s | — |
| LightGBM | 76.4s | 110.6s | 54.3% | 166s | -13.3% |
| Quantile q50 | 76.0s | 109.9s | 54.3% | 165s | -13.7% |

**Top permutation importance:** `log_eta` (66.5%) → `hour_cos` (7.1%) → `driver_avg_logratio` (5.4%) → `end_town` (3.0%) → `start_h3_te` (2.6%)

---

### Pipeline C — `final/` 自動搜尋版

A modular, config-driven pipeline with automatic greedy forward feature search and Optuna hyperparameter tuning.

```bash
# Full pipeline
python -m eta_pipeline.run --config final/config/default.yaml --work-dir .

# Quick smoke test
python -m eta_pipeline.run --quick --work-dir .

# Resume a specific phase
python -m eta_pipeline.run --config final/config/default.yaml \
    --phase final --run-id <run_id> --work-dir .
```

**Feature search result (greedy forward, 8 blocks selected from 18):**

| Step | Block added | Val MAE | Gain |
|---|---|---|---|
| Start | `base_eta` | 85.5s | — |
| 1 | `time_basic` | 80.6s | -4.8s |
| 2 | `time_cyclic` | 80.6s | -3.3s |
| 3 | `driver_bias` | 78.2s | -3.3s |
| 4 | `time_flags` | 78.2s | -1.6s |
| 5 | `region_cat` | 77.3s | -0.9s |
| 6 | `geo_raw` | 77.1s | -0.7s |
| 7 | `geo_distance` | 77.1s | -0.2s |

**Results (test: full May, 205k rows):**

| Model | MAE | RMSE | Within 60s | P90 | MAE Improvement |
|---|---|---|---|---|---|
| Baseline | 88.5s | 127.5s | 49.4% | 197s | — |
| LightGBM (Optuna 30 trials) | 75.8s | 109.3s | 54.5% | 165s | -14.3% |

Run tests:
```bash
python -m pytest final/tests/ -v
```

---

## Key Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute error in seconds |
| **RMSE** | Like MAE but penalizes large errors more — sensitive to the long tail |
| **P90** | 90th percentile of absolute error |
| **Late>60s** | % of trips where driver arrived >60s later than predicted |
| **Within60s** | % of trips predicted within 1 minute of actual |

**Priority: RMSE and P90 over MAE.** The goal is eliminating catastrophic experiences, not just improving the average.

---

## Outputs

All outputs are stored under `outputs/` and organized by pipeline run:

| Path | Content |
|---|---|
| `outputs/run_src/metrics/all_metrics.json` | Pipeline A full metrics |
| `outputs/run_src/figures/` | Pipeline A calibration, error distribution, MAE by hour |
| `outputs/run_mix/metrics/` | Pipeline B metrics + permutation importance |
| `outputs/run_mix/figures/` | Pipeline B figures incl. permutation importance plot |
| `outputs/figures_comparison/` | Side-by-side A vs B comparison figures |
| `outputs/comparison_report.md` | A vs B written comparison |
| `artifacts/selected_*.json` | Pipeline C selected feature blocks |
| `artifacts/best_params_*.json` | Pipeline C Optuna best hyperparameters |
| `reports/run_*.txt` | Pipeline C full text reports |
