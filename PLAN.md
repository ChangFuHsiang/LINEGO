# ETA-Correction Experimentation Pipeline — Plan

> **Goal.** Turn the current monolithic `eta_correction.py` script into a modular,
> config-driven experimentation pipeline that can (1) declare reusable feature
> blocks, (2) automatically search over feature-block combinations, (3) tune
> hyperparameters with Optuna, and (4) emit a single self-contained `.txt`
> analysis report per run.
>
> This document is a **design/plan only** — no pipeline code is implemented yet.

---

## 0. Context & Problem Restatement

We correct a third-party ETA. For each trip we have a provider estimate
`driver_eta` (seconds from driver-accept to arrival at pickup) and the realized
`time_accept_to_arrive`. We model the **residual**:

```
target  eta_error = time_accept_to_arrive - driver_eta
serve   corrected_eta = driver_eta + model.predict(features)
```

Modeling the residual (not the raw arrival time) keeps the provider's signal and
asks the model only to learn its systematic bias — this is the right framing and
we keep it.

### Dataset (from `trip_stats_eta.parquet)

| column | type | notes |
|---|---|---|
| `trip_id` | int | row id, **never a feature** |
| `uid_hash` | str | rider id (hashed) |
| `did_hash` | str | driver id (hashed) |
| `start_lat`, `start_lng` | float | pickup origin |
| `start_county`, `start_town` | str | admin region (Chinese), high-cardinality `town` |
| `end_lat`, `end_lng` | float | destination, **nullable** |
| `end_county`, `end_town` | str | **nullable** |
| `end_address` | str | free text, currently unused |
| `request_time` | int (unix s) | request timestamp |
| `driver_eta` | int (s) | provider estimate — the signal we correct |
| `time_accept_to_arrive` | int (s) | **target source** (accept → arrive at pickup) |
| `time_start_to_finish` | int (s) | trip leg *after* pickup — **post-outcome, leakage risk** |

### ⚠️ Leakage rules (must be enforced in code)
- **`time_accept_to_arrive`** is only allowed inside the target. Never a feature.
- **`time_start_to_finish`** happens *after* the quantity we predict — it is **not
  available at inference time**. It must be excluded from all features. (Useful only
  for offline EDA.)
- **`trip_id`, `uid_hash`, `end_address`** are excluded as model inputs (row id,
  rider id, and free text respectively — see §8 decisions). `end_address` parsing is
  explicitly out of scope; `uid_hash`/rider features are not used.
- Any **aggregate / target-encoded feature** (driver bias, demand counts, region
  target-encoding) must be **fit on the training split only** and merged into
  val/test — never fit on the row's own split.

### Splits (TBD)
- **Train:** Dec 2025, Jan, Feb, Mar 2026
- **Validation:** Apr 2026  → used for feature selection & Optuna objective
- **Test (holdout):** May 2026 → **touched exactly once**, only in the final report

Rationale: ETA bias drifts seasonally (Lunar New Year 2026-01-28→02-05 is a known
regime); a forward-chained time split is the honest estimate of deployment error
and prevents temporal leakage.

### Metrics (selection + reporting)
- **Primary (selection):** validation **MAE** of `eta_error` (matches `regression_l1`).
- **Reporting suite (on corrected_eta vs actual):** MAE, RMSE, bias (mean signed
  error), within ±60 s %, within ±120 s %, P50/P90/P95 absolute error,
  over-promise rate (`actual - corrected > 60`), under-promise rate
  (`corrected - actual > 60`), all vs the **raw `driver_eta` baseline**.

---

## 1. Objectives & Non-Goals

**Objectives**
1. **Feature registry** — feature engineering expressed as small, named, reusable
   *blocks* that can be toggled on/off by config (req. 1).
2. **Automatic feature-combination search** — search over block subsets, return a
   ranked leaderboard and a chosen set (req. 2).
3. **Automatic hyperparameter search** — Optuna (already prototyped) integrated as a
   first-class, configurable stage (req. 3).
4. **Run report** — one human-readable `.txt` per run capturing config, search
   results, final metrics, segment analysis, and reproducibility info (req. 4).

**Non-Goals (v1)**
- No online serving / API. Offline batch training + report only.
- No deep-learning models; LightGBM only (the search interface is model-agnostic so
  this can extend later).
- No `end_address` NLP parsing (parked for a future iteration).
- No distributed/multi-node execution; single machine, optional CUDA.

---

## 2. Proposed Architecture

Refactor the single script into an installable package with thin, testable modules.
A run is a 3-phase orchestration (feature search → HP tuning → final fit/report),
each phase independently skippable / resumable via config.

```
final/
├── config/
│   ├── default.yaml            # canonical config (data paths, splits, search, model)
│   └── quick.yaml              # tiny/fast config for smoke tests & CI
├── src/eta_pipeline/
│   ├── __init__.py
│   ├── config.py               # dataclasses + YAML loader + validation
│   ├── data.py                 # load, clean, null-fill, time-based split
│   ├── features.py             # FeatureBlock registry, leakage-safe fit/transform
│   ├── feature_search.py       # forward / ablation / optuna feature selection
│   ├── tuning.py               # Optuna HP study (resumable, persisted)
│   ├── model.py                # LightGBM train/predict/save wrappers
│   ├── metrics.py              # MAE/RMSE/within-X/over-under-promise + segments
│   ├── report.py               # assemble the .txt report
│   ├── plotting.py             # (optional) PNG residual/importance plots
│   └── run.py                  # CLI orchestrator (the entry point)
├── tests/                      # pytest: leakage, determinism, registry, metrics
├── reports/                    # run_<runid>.txt  (+ optional .json sidecar)
├── artifacts/                  # model_<runid>.lgb, study_<runid>.db, config_<runid>.yaml
├── eta_correction.py           # KEPT as legacy reference (or thin shim to run.py)
├── environment.yml / pyproject.toml  # extend deps: pyyaml, (optional) matplotlib
└── PLAN.md                     # this file
```

**Design principles**
- **Config is the single source of truth.** Every run is fully described by one YAML
  (snapshotted into `artifacts/config_<runid>.yaml` + echoed in the report).
- **Leakage-safe by construction.** All stateful feature blocks implement
  `fit(train_df)` then `transform(df)`; `fit` is *only* ever called on train.
- **Compute features once.** Build the **superset** of all enabled blocks a single
  time, then the combination search just **selects columns** — never recomputes. This
  is the key to making the search affordable.
- **Determinism.** Global seed threaded through numpy / LightGBM / Optuna sampler;
  library versions captured in the report.
- **Each phase is resumable.** Optuna study persisted to SQLite; phases can be
  individually disabled to re-run only what's needed.

---

## 3. Module-by-Module Design

### 3.1 `config.py`
- Dataclasses: `DataConfig`, `SplitConfig`, `FeatureSearchConfig`, `TuningConfig`,
  `ModelConfig`, `ReportConfig`, `RunConfig`.
- Load from YAML, apply defaults, **validate** (e.g. enabled blocks exist in the
  registry, split months are disjoint, metric name is known).
- Generates a `run_id` (`YYYYMMDD-HHMMSS` + short config hash) used to name all
  artifacts so concurrent/repeated runs never clobber each other.

### 3.2 `data.py`
- `load_raw(path)`: read parquet; assert expected columns present.
- `clean(df)`: parse `request_time` → datetime; null-fill (`end_lat/lng` ←
  `start_lat/lng`; `end_county/town` ← `start_*`; remaining region nulls →
  `"unknown"`), as the current script does. Centralize all null policy here.
- `add_target(df)`: `eta_error = time_accept_to_arrive - driver_eta`.
- `make_splits(df, SplitConfig)`: return `train/val/test` by month membership;
  log row counts and date ranges; assert non-empty and disjoint.
- **Data-audit hook** (runs once, logged into report): % nulls per column,
  `driver_eta`/`target` distributions & clip bounds, count of `end_*` nulls,
  detection of the LNY window, sanity range checks (non-negative times, plausible
  lat/lng for Taiwan). Optionally **clip/winsorize** extreme `eta_error` for training
  robustness (configurable; report the clip rule).

### 3.3 `features.py` — the feature registry (req. 1)

The mechanism is fixed; the feature *content* is open. We propose organizing feature
engineering as small, named **blocks** — each a unit that produces one or more
columns and declares which are categorical. Stateless blocks transform row-wise;
stateful blocks learn aggregates on train. A sketch of the interface (exact shape is
itself a suggestion, to be refined in implementation):

```python
class FeatureBlock:
    name: str                      # registry key, e.g. "driver_bias"
    output_cols: list[str]
    categorical_cols: list[str]    # subset of output_cols
    depends_on: list[str] = []     # other blocks/raw cols required first
    cost: str = "cheap"            # "cheap" | "expensive" (for search ordering)

    def fit(self, train_df) -> None: ...     # train-only; no-op for stateless
    def transform(self, df) -> pd.DataFrame: ...   # returns just output_cols
```

A central `REGISTRY: dict[str, FeatureBlock]` + a `build_features(blocks, splits)`
driver that: topologically orders by `depends_on`, calls `fit` on **train only**,
then `transform` on each split, and concatenates. Output also yields the
`categorical_feature` list LightGBM needs.

#### Suggested feature blocks (ideas to start from — not a required set)

The blocks below are **suggestions** meant to spark ideas, not a checklist. Treat
them as a menu: keep what earns its place in the search, drop what doesn't, and add
your own. The `cat?` / `stateful` annotations are guidance for whoever implements a
given block. We've kept it broader than today's hardcoded `FEATURES` precisely so the
search in §3.4 has room to explore.

| block name | output columns | cat? | stateful | idea / rationale |
| --- | --- | --- | --- | --- |
| `base_eta` | `driver_eta` | no | no | the signal we correct; a natural baseline to keep |
| `time_basic` | `hour`, `day_of_week`, `is_weekend`, `month`, `day_of_month` | no | no | plain calendar features |
| `time_cyclic` | `hour_sin/cos`, `dow_sin/cos` | no | no | smooth periodicity, if linear hour hurts |
| `time_flags` | `is_rush_hour`, `is_night`, `is_lunar_new_year` | no | no | regime flags; LNY window |
| `geo_raw` | `start_lat/lng`, `end_lat/lng` | no | no | raw coords (LGBM can split on them) |
| `geo_distance` | `haversine_m`, `lat_delta`, `lng_delta`, `bearing` | no | no | trip geometry |
| `geo_center` | `dist_from_city_center_start/end` | no | no | proxy for downtown congestion |
| `geo_flags` | `same_county`, `same_town`, `is_cross_county` | int flags | no | intra vs inter-region |
| `region_cat` | `start_county/town`, `end_county/town` | yes | no | native LGBM categoricals |
| `region_freq` | `*_town_freq`, `*_county_freq` | no | yes | frequency encoding (train counts) |
| `region_target_enc` | `start_town_te`, `end_town_te` | no | yes | smoothed, out-of-fold target enc of `eta_error` |
| `driver_bias` | `driver_avg_error`, `driver_median_error`, `driver_error_std` | no | yes | per-`did_hash`, excl. LNY, smoothed by trip count |
| `driver_experience` | `driver_trip_count` | no | yes | volume / reliability proxy |
| `dest_demand` | `dest_demand_score` | no | yes | trips→`end_town` (current feature) |
| `origin_demand` | `origin_demand_score` | no | yes | trips from `start_town` |
| `od_pair_stats` | `od_count`, `od_mean_error` | no | yes | (`start_town`→`end_town`) aggregates, smoothed |
| `eta_derived` | `eta_per_km`, `log_driver_eta`, `eta_x_rush` | no | no | implied speed & interactions |
| `demand_by_hour` | `dest_demand_hour` | no | yes | town×hour demand (sparser) |

**Decided exclusions** (see §8): no **rider / `uid_hash`** features (sparse, weak
signal vs. driver-side bias) and no **`end_address`** parsing — both out of scope.
Other directions worth exploring later, if the above plateaus: weather joins, holiday
calendars beyond LNY, or richer driver×region interaction stats. None of these are
commitments — they're prompts.

#### One genuine constraint: stateful blocks must stay leakage-safe

This is the part that is *not* optional, because getting it wrong silently inflates
the scores. Whatever stateful blocks you end up writing, they should follow:

- Aggregations computed **on train only**, merged left into each split.
- **Smoothing** for low-support groups, e.g.
  `enc = (n·group_mean + m·global_mean) / (n + m)` with a configurable prior weight `m`.
- **Target encodings** use **K-fold out-of-fold** encoding *within train* to avoid the
  model fitting its own labels; val/test use full-train encoding.
- `driver_bias` (and similar) should exclude the LNY window from the mean, as the
  current code does.
- Unseen groups in val/test fall back to global mean / 0 (decide and document per block).

### 3.4 `feature_search.py` — automatic combination search (req. 2)

Search operates over the set of **enabled blocks** declared in config. Strategies
(selectable; default = greedy forward):

1. **`forward`** (greedy forward selection) — start from a mandatory core
   (`base_eta`), repeatedly add the single block that most improves val MAE; stop when
   no block improves by more than `min_gain` or `max_blocks` reached. *O(k²)* fits,
   good default.
2. **`backward`** (recursive elimination) — start from all blocks, drop the block
   whose removal least hurts (or helps) val MAE. Complements forward.
3. **`ablation`** (leave-one-block-out) — fit full set once, then one fit per block
   removed; reports each block's marginal contribution. Great for the report even if
   not used for selection.
4. **`exhaustive`** — all `2^k` subsets; only allowed when `k ≤ exhaustive_max`
   (guard, default 6) to avoid blow-up.
5. **`optuna_select`** — treat each block as a boolean `suggest_categorical` and let
   a TPE study minimize val MAE; can be **fused with HP tuning** (see §3.5 joint mode).

**Affordability levers (all config-driven):**
- Build the **feature superset once**, cache it; each candidate = a column slice.
- Use **proxy training settings** during search (fixed reasonable HP, capped
  `num_boost_round` with early stopping, e.g. lr=0.05 / 300 rounds) — full HP tuning
  happens *after* the feature set is fixed.
- Optional **row subsample** of train for search (`search_sample_frac`), with the
  final fit always on full data.
- A small **in-memory + on-disk cache** keyed by `frozenset(blocks)+params_hash` so
  repeated subsets are never refit.

**Output:** a ranked leaderboard (block-set, val MAE, Δ vs core, #features, fit time)
→ stored for the report; the winning block set is passed to the tuning phase.

### 3.5 `tuning.py` — hyperparameter search (req. 3, extends current Optuna)

- Wrap the existing objective. Search space identical to current script
  (`learning_rate`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`,
  `bagging_fraction`, `bagging_freq`, `lambda_l1/l2`) but **fully config-driven**
  (ranges/log-scale in YAML so codex/users can adjust without code edits).
- `objective(trial)` trains on the **chosen feature set** with early stopping on val,
  returns val MAE, records `best_iteration` as a trial user-attr.
  - For the pruner to function, the objective must report the intermediate val metric
    per boosting round and honor prune signals — wire
    `optuna.integration.LightGBMPruningCallback(trial, "l1", valid_name=...)` into
    `lgb.train`'s callbacks (alongside `early_stopping`). Without this callback,
    `MedianPruner` has nothing to act on and silently does nothing.
- **Persisted, resumable study**: `optuna.create_study(storage="sqlite:///artifacts/
  study_<runid>.db", study_name=runid, load_if_exists=True)`, `TPESampler(seed)`,
  `MedianPruner` **on by default** (`n_startup_trials` warmup) to kill weak trials
  early — this is what keeps a generous `n_trials: 150` cheap. Resuming requires an explicit
  `--run-id` (see §3.10b); reads the chosen feature set from `selected_<runid>.json`.
- Honor `n_trials` **and** `timeout` (current script uses 50 / 7200 s).
- Capture **Optuna hyperparameter importances** (`get_param_importances`) for the report.

**Joint mode (optional `search.mode: joint`):** a single Optuna study suggests both
block-inclusion booleans *and* HP in one objective. More powerful but costlier and
harder to interpret.

**Decided (see §8): default is `sequential`.** Not for cost reasons — on GPU the
joint search would actually be affordable — but because sequential produces the
per-block leaderboard + ablation table the report (req. 4) depends on, and is easier
to debug/reproduce. `joint` stays available as an opt-in for users who want it.

### 3.6 `model.py`
- `train(features, label, params, cat_features, num_boost_round, valid_sets,
  early_stopping)` → returns model + `best_iteration`. Single place that sets
  `device` (config: `cuda` with **graceful CPU fallback** if CUDA init fails —
  important since the env builds LightGBM-CUDA but other machines may not have it; see
  §3.10g for the detection mechanism).
- `predict`, `save_model`, `load_model`, `feature_importance(type="gain"|"split")`.
- **Final-fit policy (kept from current script):** retrain on **train+val** combined
  using `int(best_iteration * round_inflation)` rounds (default 1.1), then evaluate on
  May test. Document this in the report.

### 3.7 `metrics.py`
- Core: `mae`, `rmse`, `bias`, `within(threshold)`, `over_promise(tol)`,
  `under_promise(tol)`, `abs_error_percentiles([50,90,95])`.
- `evaluate(actual, corrected, raw)` → dict comparing corrected vs raw baseline.
- **Segment analysis:** `evaluate_by(df, by=...)` for slices — by `hour`,
  `day_of_week`, `start_county`, distance bucket, `is_lunar_new_year`,
  `is_rush_hour`, and `driver_eta` magnitude bucket. Surfaces where the model helps /
  hurts (key analytical value). Default bucket edges (config-overridable):
  distance `[0, 2, 5, 10, 20, ∞] km`, `driver_eta` `[0, 120, 300, 600, ∞] s`; buckets
  with fewer than `min_segment_rows` (default 200) are folded into an "other" row so
  noisy slices don't dominate the report.

### 3.8 `report.py` — the `.txt` analysis output (req. 4)

Writes `reports/run_<runid>.txt` (plus an optional machine-readable
`reports/run_<runid>.json` sidecar for downstream tooling). Plain text, fixed-width
tables, section banners. Proposed layout:

```
================================================================
 ETA CORRECTION — RUN REPORT
================================================================
Run id            : 20260529-143000-a1b2c3
Timestamp / host  : 2026-05-29 14:30:00  /  <hostname>
Git commit        : <sha or "n/a">
Library versions  : python / lightgbm / optuna / numpy / pandas / sklearn
Device            : cuda (fallback: cpu) — used: cuda
Random seed       : 42

---------------- 1. DATA & SPLITS ------------------------------
Rows total / train / val / test  + date ranges per split
Null %, target & driver_eta distributions, clip rule, LNY rows

---------------- 2. CONFIG SNAPSHOT ----------------------------
Enabled blocks, search strategy + budget, tuning ranges + budget

---------------- 3. FEATURE SEARCH -----------------------------
Strategy = forward. Selection trace (block added → val MAE).
Final block set + resulting feature columns.
Leaderboard (top-N subsets): blocks | val MAE | Δcore | #feat | t(s)
Ablation table (marginal contribution per block), if computed.

---------------- 4. HYPERPARAMETER TUNING ----------------------
#trials, best val MAE, best_iteration, best params (pretty).
Optuna param-importance ranking. (Top/worst trials summary.)

---------------- 5. FINAL MODEL — MAY HOLDOUT ------------------
                         Raw API      Corrected     Delta
MAE (s) / RMSE (s) / Bias (s)
Within ±60s % / ±120s % / P50 / P90 / P95 abs err
Over-promise % / Under-promise %  (corrected vs raw)

---------------- 6. SEGMENT ANALYSIS ---------------------------
MAE raw vs corrected by hour / county / distance bucket / rush /
weekend / LNY  — highlight regressions (segments where corrected > raw).

---------------- 7. FEATURE IMPORTANCE -------------------------
Gain & split importance table (sorted).

---------------- 8. ARTIFACTS & REPRODUCIBILITY ---------------
model path, study db path, config snapshot path, exact CLI to reproduce.
================================================================
```

`plotting.py` (optional) can dump residual histogram, error-vs-distance, and
importance bar charts as PNGs into `reports/` — flagged off by default to keep the
core deliverable text-only.

### 3.9 `run.py` — orchestrator / CLI
```
python -m eta_pipeline.run --config config/default.yaml \
    [--phase all|feature|tune|final] [--run-id NAME] [--seed 42] [--quick]
```
Flow: load+validate config → load/clean/split data + audit → build feature superset
(fit-on-train) → **Phase A** feature search → **Phase B** HP tuning on chosen set →
**Phase C** final retrain (train+val) + May eval + segment analysis → write report +
save artifacts. Each phase guarded so a run can resume/skip. All stdout also mirrored
into the report so nothing is lost.

### 3.10 Cross-cutting contracts (read before implementing)

These are the decisions that aren't owned by any single module but, if left
implicit, cause divergent implementations or silent bugs. Treat them as binding.

#### (a) Inter-phase handoff artifacts

Phases communicate **only through files in `artifacts/`**, keyed by `run_id`, so any
phase can run standalone (`--phase tune`) by reading the previous phase's output.
This is what makes "skippable / resumable phases" actually work.

| phase | writes | reads |
| --- | --- | --- |
| feature (A) | `selected_<runid>.json` = `{"blocks": [...], "columns": [...], "categorical": [...]}` + the leaderboard | — |
| tune (B) | `best_params_<runid>.json` = `{"params": {...}, "best_iteration": N}` | `selected_<runid>.json` |
| final (C) | `model_<runid>.lgb`, the report, `run_<runid>.json` sidecar | `selected_<runid>.json`, `best_params_<runid>.json` |

If a required input file is missing when a phase is run in isolation, fail fast with
a clear message ("run --phase feature first, or use a config with that phase
enabled"). When phases run together in one invocation they pass objects in memory
**and** still write these files (so a later isolated phase can resume).

#### (b) `run_id` & resume semantics

- Default `run_id` = `YYYYMMDD-HHMMSS-<config_hash8>`; printed at startup and into the
  report. Because of the timestamp, two default invocations never collide.
- **To resume / re-run a single phase you MUST pass `--run-id <id>` explicitly** so
  the new invocation reads the prior artifacts and the same Optuna SQLite study
  (`study_<runid>.db`, opened with `load_if_exists=True`). State this in `--help`.

#### (c) `build_features` return type & row alignment

`build_features(blocks, splits, config)` returns a dict keyed by split name:

```python
{
  "train": FeatureBundle(X: pd.DataFrame, y: pd.Series, cat_cols: list[str]),
  "val":   FeatureBundle(...),
  "test":  FeatureBundle(...),
}
```

- `X` carries **all** superset columns; the search selects a sublist per candidate,
  and the per-candidate `cat_cols` is just `set(candidate_cols) & cat_cols`.
- **Index discipline:** every block's `transform` must return a frame indexed
  identically to its input split (no `reset_index` mid-pipeline). `X`, `y`, and the
  raw split frame share one index so segment analysis can join back by index.

#### (d) Categorical handling across splits (prevents a silent accuracy bug)

Pandas assigns category **codes per frame**, so naively doing `.astype("category")`
on each split independently gives train/val/test *different* code→label maps and
LightGBM then trains and predicts on misaligned integers.

- Category dtypes are **fixed on train** and applied to val/test:
  `df[col] = pd.Categorical(df[col], categories=train_categories[col])`.
- Values unseen in train (or nulls) map to the explicit `"unknown"` category, which
  must itself be present in `train_categories`.
- The `train_categories` mapping is part of the fitted feature state and is reused by
  the final-fit phase (train+val) as well.

#### (e) Subsample-vs-aggregate rule during search

When `search_sample_frac < 1.0`, the subsample affects **only the rows fed to
`lgb.train` during the search**. Stateful aggregates (driver bias, demand, target
encoding) are still **fit on full train** and merged — i.e. we never recompute
feature state on the subsample. This keeps candidate scores comparable and avoids
re-fitting aggregates per candidate. The final-fit phase always uses full data.

#### (f) OOF target-encoding determinism

`region_target_enc` (and any target-encoded block): K folds (`target_enc_folds`)
assigned by **`KFold(shuffle=True, random_state=run.seed)`** over the train rows
(group-agnostic is acceptable for v1; note as a future refinement that grouped/time
folds reduce leakage further). val/test use the full-train encoding. The fold seed is
derived from the run seed so encodings are reproducible.

#### (g) CUDA fallback detection

`device: cuda` is attempted by training a 1-tree model on a tiny slice inside a
`try/except` at startup; on any exception, log a warning, set `device="cpu"`, and
record both the requested and actual device in the report (the report header already
has a "requested / used" field).

---

## 4. Configuration Example (`config/default.yaml`)

```yaml
run:
  seed: 42
  device: cuda            # cpu fallback automatic
  phases: [feature, tune, final]

data:
  path: trip_stats_eta.parquet
  clip_target_quantiles: [0.001, 0.999]   # null to disable

split:
  train_months: [12, 1, 2, 3]
  val_months:   [4]
  test_months:  [5]
  lny_window:   ["2026-01-28", "2026-02-05"]

features:
  core_blocks: [base_eta]                  # always included
  candidate_blocks:                        # the search space
    [time_basic, time_cyclic, time_flags, geo_distance, geo_center,
     geo_flags, region_cat, region_freq, region_target_enc,
     driver_bias, driver_experience, dest_demand, origin_demand,
     od_pair_stats, eta_derived]
  target_enc_smoothing: 100
  target_enc_folds: 5

search:
  mode: sequential          # sequential | joint
  strategy: forward         # forward | backward | ablation | exhaustive | optuna_select
  min_gain_seconds: 0.2
  max_blocks: 12
  exhaustive_max: 6
  search_sample_frac: 1.0
  proxy_params: {learning_rate: 0.05, num_boost_round: 400, early_stopping: 50}
  also_run_ablation: true   # for the report

tuning:
  n_trials: 150           # generous — GPU run is fast (Q4); timeout is the real cap
  timeout_seconds: 7200   # binding safety cap; stop here even if n_trials not reached
  num_boost_round: 2000
  early_stopping: 50
  round_inflation: 1.1
  pruner: median          # MedianPruner — kill weak trials early so extra trials are cheap
  n_startup_trials: 10    # quasi-random warmup before TPE/pruner engage
  space:
    learning_rate:   {low: 0.01, high: 0.1, log: true}
    num_leaves:      {low: 32,   high: 256}
    min_data_in_leaf:{low: 50,   high: 500}
    feature_fraction:{low: 0.6,  high: 1.0}
    bagging_fraction:{low: 0.6,  high: 1.0}
    bagging_freq:    {low: 1,    high: 10}
    lambda_l1:       {low: 1e-4, high: 10.0, log: true}
    lambda_l2:       {low: 1e-4, high: 10.0, log: true}

model:
  objective: regression_l1   # Q1 OPEN: switch to `quantile` (alpha ~0.55-0.6) to bias
  metric: mae                # toward conservative ETAs without changing the selection metric
  # quantile_alpha: 0.58     # used only when objective: quantile

report:
  dir: reports
  emit_json_sidecar: true
  emit_plots: false
  min_segment_rows: 200                       # smaller slices folded into "other"
  distance_buckets_km: [0, 2, 5, 10, 20]      # last bucket is [20, ∞)
  driver_eta_buckets_s: [0, 120, 300, 600]    # last bucket is [600, ∞)
```

> Note: `config/default.yaml` above is illustrative. The implementer should keep the
> dataclasses in `config.py` and this file in sync, and the validator should reject
> unknown keys so the config never silently drifts from the code.

---

## 5. Risks, Edge Cases & Mitigations

| risk | mitigation |
|---|---|
| **Target leakage** via `time_start_to_finish` / aggregates fit on wrong split | hard-coded feature deny-list; stateful blocks only `fit(train)`; OOF target encoding; unit tests assert no forbidden column reaches the model |
| **Feature-search overfits to Apr val** | keep May untouched until final; report Δ between val and test; prefer simpler block sets on near-ties (parsimony tie-break) |
| **Combinatorial blow-up** | search over *blocks* not columns; `exhaustive_max` guard; greedy default; superset-compute-once + caching |
| **High-cardinality `town`** | native LGBM categorical + frequency/smoothed-target encodings; `unknown` bucket for nulls/unseen |
| **CUDA unavailable on a machine** | `device` config with automatic CPU fallback + warning in report |
| **Distribution shift (LNY, seasonality)** | LNY flag + exclude LNY from driver bias; segment report by LNY/rush; time-based split already models drift |
| **Non-determinism across runs** | single seed threaded to numpy/LGBM/Optuna; versions + seed logged; force `deterministic`/single-thread option for CI |
| **Long runtime** | `--quick` config (few trials, subsample, few blocks) for smoke tests; timeout honored; phases resumable |
| **Outliers in `eta_error`** | optional winsorization (config); `regression_l1` is already robust; report raw vs clipped effect |

---

## 6. Testing & Validation Plan

- **Unit:** registry resolves dependencies/topo-order; each stateful block fits only on
  train (assert no val rows touched); metric functions vs hand-computed values;
  config validation rejects bad blocks/overlapping months.
- **Leakage test:** assert `time_accept_to_arrive`, `time_start_to_finish`, `trip_id`,
  `end_address` never appear in the model's feature columns; assert a target-encoded
  column computed with OOF differs from naive in-fold encoding.
- **Determinism test:** two runs with same seed + config → identical val MAE.
- **Smoke test (CI):** `config/quick.yaml` (subsample, 2–3 blocks, 3 Optuna trials)
  runs end-to-end and produces a report in < ~1 min.
- **Regression guard:** the legacy `FEATURES` set reproduces the current script's
  ballpark May MAE — sanity that the refactor didn't change behavior.

---

## 7. Implementation Roadmap (incremental, each step runnable)

1. **Scaffold + config** — package layout, `config.py`, `default.yaml`, `quick.yaml`;
   port deps into `environment.yml`/`pyproject.toml` (add `pyyaml`).
2. **Data module** — `data.py` (load/clean/split/audit), reproducing current
   preprocessing exactly; wire the data-audit log.
3. **Feature registry** — `features.py` with all blocks + leakage-safe fit/transform +
   superset builder + caching; unit + leakage tests.
4. **Model + metrics** — `model.py`, `metrics.py` (incl. segment analysis); verify
   parity with the legacy script on the legacy feature set.
5. **Feature search** — `feature_search.py` (forward + ablation first; others behind
   config); leaderboard output.
6. **Tuning** — `tuning.py` (config-driven space, persisted/resumable study,
   param-importance); optional joint mode.
7. **Report** — `report.py` (+ optional `plotting.py`) assembling all sections; JSON
   sidecar.
8. **Orchestrator + CLI** — `run.py` tying phases together with skip/resume; mirror
   stdout into report.
9. **CI smoke test + docs** — `quick.yaml` end-to-end run; short README usage section;
   keep `eta_correction.py` as a thin shim or reference.

**Suggested first deliverable for review:** steps 1–4 (scaffold, data, registry,
model/metrics) — establishes the leakage-safe foundation everything else builds on.

---

## 8. Note

- **Splits are currently marked `TBD` in §0.** The plan assumes the existing
  Dec–Mar / Apr / May time-based split. If you want to revisit (e.g. different month
  boundaries, or rolling-origin CV instead of a single val month), flag it — it
  changes `data.make_splits`, the leakage reasoning, and the report's split section.
