# LINEGO — ETA Correction Pipeline

A machine learning pipeline that corrects systematic underestimation in a ride-hailing platform's driver arrival time (ETA) predictions.

---

## The Problem

The platform's routing API provides `driver_eta` — how many seconds until the driver reaches the passenger pickup point. This estimate is consistently **too optimistic**: drivers arrive ~76 seconds later than promised on average. This causes two problems:

1. Dispatch may select the wrong driver
2. Users see "3 minutes" but wait 6 minutes

---

## The Approach: Residual Learning

Instead of rebuilding a routing engine from scratch, the model learns **how wrong the API is**, then corrects it:

```
corrected ETA = driver_eta × exp(model_prediction)
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
```

---

## How to Run

Run each script in order from the project root:

```bash
python src/load_clean.py    # Step 1: clean data
python src/features.py      # Step 2: build features
python src/train.py         # Step 3: train models
python src/evaluate.py      # Step 4: generate charts & metrics
```

Each step saves its output as a `.parquet` file so any step can be re-run independently without reprocessing everything from scratch.

---

## Pipeline Walkthrough

### `config.py` — Settings

Defines all constants: file paths, train/valid/test date splits, LightGBM hyperparameters, target encoding smoothing strengths, and a hardcoded list of Taiwan public holidays for 2026.

**Data splits (chronological, no shuffling):**
- Train: Jan 1 – Apr 22
- Validation: Apr 23 – May 6
- Test: May 7 – May 20

---

### `load_clean.py` — Clean the Raw Data

**Input:** `data/raw/trip_stats_eta.parquet`  
**Output:** `data/trips_clean.parquet`

1. **Removes bad rows** — trips where `driver_eta <= 0` or `time_accept_to_arrive <= 0`, and 17 rows missing county info
2. **Converts timestamps** — `request_time` is UTC Unix seconds; adds 8 hours to get Taiwan local time
3. **Removes extreme outliers** — drops trips above the 99.9th percentile of actual arrival time
4. **Computes two target variables** (what the model predicts):
   - `target_logratio = log(actual / driver_eta)` — the log-ratio of how wrong the API was. Preferred because errors are proportional: a 2-minute delay on a 10-minute trip is very different from a 2-minute delay on a 2-minute trip.
   - `target_additive = actual - driver_eta` — raw difference in seconds (simpler alternative)

---

### `features.py` — Build Features

**Input:** `data/trips_clean.parquet`  
**Output:** `data/feats.parquet`

Features are the signals the model uses to detect *when and where* the API tends to be wrong.

**H3 hexagonal grid**

Raw GPS coordinates are unique per trip — the model can't learn from a single point. The H3 library groups nearby coordinates into hexagonal cells:
- Resolution 9 (~170m): captures fine-grained spots like "this corner is always hard to find"
- Resolution 8 (~500m): broader neighborhood-level backup

**Time features**

| Feature | Description |
|---|---|
| `hour_sin` / `hour_cos` | Hour encoded as a circle so 11pm and midnight are treated as close together |
| `dow` | Day of week (1=Mon … 7=Sun) |
| `is_weekend` | 1 if Saturday or Sunday |
| `log_eta` | The API's own estimate (log-transformed) — the most important single feature |
| `is_special_day` | 1 if the date is a Taiwan public holiday |

**Target encoding**

Categorical features (driver ID, H3 cell, district) are converted to numbers by replacing each category with its **historical average error**. For example, a driver who consistently arrives late gets a higher encoded value.

To prevent data leakage (the model "cheating" by seeing its own answers):
- **Validation/test sets:** encoded using statistics computed from training data only
- **Training set itself:** uses **5-fold Out-Of-Fold (OOF)** — each training row is encoded using statistics from the *other 4 folds*, never its own

**Demand proxy**

For each (district × hour) combination, counts historical trip volume. Low demand = fewer drivers nearby = harder to reach = larger ETA error.

---

### `train.py` — Train the Models

**Input:** `data/feats.parquet`  
**Output:** `outputs/models/`, `outputs/metrics/all_metrics.json`

Three models are trained and compared:

**Baseline**  
Use `driver_eta` directly with no correction. This is the "do nothing" benchmark (~MAE 123s, RMSE 246s).

**LightGBM point-estimate model**  
Gradient boosting — builds hundreds of small decision trees where each tree corrects the mistakes of the previous one. Handles non-linear patterns and feature interactions automatically. Uses early stopping (stops if validation score doesn't improve for 50 rounds).

Prediction restoration:
```python
corrected_eta = driver_eta * exp(prediction)    # log-ratio target
corrected_eta = max(0, driver_eta + prediction) # additive target
```

**Quantile models (q=0.1, 0.5, 0.9)**  
Instead of predicting the average, these predict specific percentiles:
- `q0.1` — optimistic: actual time exceeds this 90% of the time
- `q0.5` — median estimate
- `q0.9` — conservative ("better early than late"): actual time is below this 90% of the time

Showing users the q90 prediction means they are unlikely to wait longer than displayed. The three quantile outputs are sorted after prediction to prevent crossing (e.g., q90 < q50).

---

### `evaluate.py` — Visualize & Report

**Input:** `data/feats.parquet` + saved models  
**Output:** `outputs/figures/`, `outputs/metrics/`

| Output | Description |
|---|---|
| `calibration.png` | Predicted vs actual ETA — good models hug the 45° diagonal |
| `error_distribution.png` | Histogram of errors for baseline vs LightGBM |
| `mae_by_hour.png` | MAE by hour of day — identifies rush-hour problems |
| `feature_importance.png` | Which features the model relies on most |
| `quantile_coverage.png` | % of actual times falling inside the [q10, q90] interval (target: ~80%) |
| `slice_by_town.csv` | Per-district MAE — shows which areas are hardest to predict |

---

## Key Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute error in seconds |
| **RMSE** | Like MAE but penalizes large errors more — sensitive to the long tail |
| **P90** | 90th percentile of absolute error — how bad is the worst 10%? |
| **Late>60s** | % of trips where driver arrived >60s later than predicted (bad for users) |
| **Within60s** | % of trips predicted within 1 minute of actual |

**Priority: RMSE and P90 over MAE.** The goal is eliminating catastrophic experiences (waiting 5+ extra minutes), not just improving the average.

---

## Project Structure

```
LINEGO/
├── environment.yaml
├── data/
│   ├── raw/trip_stats_eta.parquet   # original data (do not modify)
│   ├── trips_clean.parquet          # output of load_clean.py
│   └── feats.parquet                # output of features.py
├── outputs/
│   ├── models/                      # trained model files (.pkl)
│   ├── metrics/                     # evaluation results (.json, .csv)
│   └── figures/                     # charts (.png)
└── src/
    ├── config.py
    ├── load_clean.py
    ├── features.py
    ├── train.py
    └── evaluate.py
```
