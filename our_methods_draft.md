# ETA 校正方法 — 完整 Draft

> 本文件涵蓋四條 pipeline：Pipeline A/B/C（ChangFu 實作，LightGBM-based）與 Pipeline D（DoDo 實作，CatBoost-based）。
> 所有 pipeline 均採**殘差學習（residual learning）**，在相同的原始資料上以不同方式建模 API 的系統性偏差。
> Draft 版本，組員可依內容調整細節與措辭。

---

## 整體方法總覽

| | A — src | B — mix | C — final | D — DoDo |
|---|---|---|---|---|
| **模型** | LightGBM | LightGBM | LightGBM | CatBoost |
| **Target** | log-ratio | log-ratio | 加法殘差 | 加法殘差 |
| **特徵選擇** | 手動固定 | 手動固定（強化） | 自動 greedy search | 手動設計（豐富） |
| **超參數調整** | 無 | 借用 C | Optuna 30 trials | — |
| **司機特徵** | 1 個 TE | 3 個統計量 | 3 個統計量 | TE + 多維交互 |
| **目的地特徵** | 無 | end_town | end_town/county/lat/lng | end_county/town + OD |
| **天氣特徵** | 無 | 無 | 無 | 有（CWA 開放資料） |
| **後處理校正** | 無 | 無 | 無 | Post-calibration |

---

---

# Pipeline A — src 純淨版

## Overall Pipeline

```
Data Input → Data Cleaning → Time-based Split → Feature Engineering
→ LightGBM (L2) + Quantile q10/q50/q90 → Evaluation
```

## 1. Data Input

使用 `trip_stats_eta.parquet` 作為唯一訓練資料，無外部輔助資料。

主要欄位：

- `driver_eta`：API 原始預估司機抵達上車點的時間（秒）
- `time_accept_to_arrive`：司機實際抵達上車點的時間，作為 ground truth
- `request_time`：叫車時間（Unix 秒，UTC），需 +8h 轉台灣當地時間
- `start_lat`, `start_lng`, `start_county`, `start_town`：上車地點
- `did_hash`：司機匿名 ID，用來學習司機個人系統性偏差
- `uid_hash`：乘客匿名 ID，用於 target encoding（效果有限）

目的地欄位（`end_*`）Pipeline A 完全不使用，原因是 2.8% 的缺失率且對上車段 ETA 的直接貢獻不明確。

## 2. Data Cleaning

- 移除 `driver_eta <= 0` 或 `time_accept_to_arrive <= 0` 的異常資料
- 移除 `start_county` 或 `start_town` 為 null 的資料
- **P99.9 上限截尾**：`time_accept_to_arrive >= 1380s` 或 `driver_eta >= 1050s` 的行程視為極端值移除
- 清洗後剩餘 **1,462,517 筆**（原始 1,546,557 筆）

Leakage control：

- `time_accept_to_arrive` 只用來建立 target，不作為 model input
- `time_start_to_finish` 完全排除（上車後才知道，deployment 時不可用）
- 所有歷史統計（TE、demand）只使用 train split 計算後套到 valid/test

## 3. Time-based Split

| Split | 日期範圍 | 筆數 | 用途 |
|---|---|---|---|
| Train | 2026-01-01 ~ 2026-04-22 | 1,166,848 | 訓練模型、計算 TE |
| Valid | 2026-04-23 ~ 2026-05-06 | 151,064 | Early stopping |
| Test | 2026-05-07 ~ 2026-05-20 | 144,605 | 最終評估 |

採用時間連續切分，避免未來資訊洩漏。

## 4. Feature Engineering

特徵總數：11 個

| 類型 | 特徵 | 說明 |
|---|---|---|
| ETA | `log_eta` | `log(driver_eta)`，對稱化數值分布 |
| Time | `hour_sin`, `hour_cos` | 時段週期特徵（讓 23 時和 0 時相近） |
| Time | `dow` | 星期幾（整數 1–7） |
| Time | `is_weekend` | 是否週末 |
| Time | `is_special_day` | 是否台灣 2026 年特殊假日（春節、清明等） |
| Location | `start_h3_te` | H3 res-9（~170m）格子的 OOF target encoding |
| Location | `start_town_te` | 行政區的 OOF target encoding |
| Driver | `did_hash_te` | 司機 ID 的 OOF target encoding |
| Rider | `uid_hash_te` | 乘客 ID 的 OOF target encoding（平滑強度高） |
| Demand | `demand_town_hour` | 同行政區同小時的歷史叫車量（供需代理） |

Target encoding 設計：valid/test 用全 train 統計；train 自身用 5-fold OOF 防洩漏。

**Target 定義（log-ratio）：**

```
target = log(time_accept_to_arrive / driver_eta)
corrected_eta = driver_eta × exp(pred)
```

選擇 log-ratio 的理由：ETA 誤差通常是乘法性的，log-ratio 自然處理這個比例關係，對長尾誤差更穩定。

## 5. Model Method

**主模型：LightGBM（L2/MSE 目標）**

- L2 損失函數：最小化平方誤差，等效於預測條件**平均值**
- Valid set early stopping（patience=50）
- 預設超參數，未進行系統性調參

**分位數模型：Quantile LightGBM（q10 / q50 / q90）**

- q50 等效 MAE 損失，預測條件**中位數**，對離群值更穩健
- q90 作為「保守估計」（寧早勿晚策略）
- 預測交叉時以排序修正（確保 q10 ≤ q50 ≤ q90）

## 6. Results

| 模型 | MAE↓ | RMSE↓ | MeanErr | Within60↑ | Within120↑ | P90↓ | Late>60↓ | MAE 改善 |
|---|---|---|---|---|---|---|---|---|
| Baseline (driver_eta) | 88.1s | 128.0s | -39.5s | 49.6% | 76.0% | 196s | 14.7% | — |
| **LightGBM (L2)** | **76.6s** | **111.1s** | -20.5s | 54.2% | 81.3% | 167s | 18.3% | **-13.2%** |
| **Quantile q50** | **76.1s** | **110.2s** | -15.8s | 54.2% | 81.7% | 165s | 19.7% | **-13.6%** |
| Quantile q90（保守） | 143.2s | 169.3s | +124.3s | 17.5% | 43.2% | 252s | 77.5% | — |

Quantile [q10, q90] 覆蓋率：**79.7%**（目標 80%，校準良好）

---

---

# Pipeline B — mix 強化版

## Overall Pipeline

```
Data Input → Data Cleaning → Time-based Split → Feature Engineering（強化）
→ LightGBM（固定最佳超參數）+ Quantile q10/q50/q90
→ Permutation Importance → Evaluation
```

## 1. Data Input

同 Pipeline A，新增使用目的地欄位：

- `end_town`：下車行政區，約 2.8%（43,727 筆）缺失 → 填入 `"unknown"` 作為獨立 category

填 `"unknown"` 而非刪除，讓 LightGBM 學習「不知道目的地」的系統性偏差。

## 2. Data Cleaning

與 Pipeline A 完全相同（1,462,517 筆）。

## 3. Time-based Split

與 Pipeline A 完全相同（共同口徑）。

## 4. Feature Engineering

特徵總數：15 個（移除 2 個、新增 6 個）

| 類型 | 特徵 | 說明 | 相對 A 的變化 |
|---|---|---|---|
| ETA | `log_eta` | 同 A | 不變 |
| Time | `hour_sin`, `hour_cos`, `dow`, `is_weekend`, `is_special_day` | 同 A | 不變 |
| **Driver（新）** | `driver_avg_logratio` | 司機 log-ratio 均值（排除春節） | 取代 `did_hash_te` |
| **Driver（新）** | `driver_median_logratio` | 司機 log-ratio 中位數 | 取代 `did_hash_te` |
| **Driver（新）** | `driver_std_logratio` | 司機 log-ratio 標準差（穩定度） | 取代 `did_hash_te` |
| **Rider（新）** | `uid_trip_count_train` | 乘客在訓練集累積叫車次數 | 取代 `uid_hash_te` |
| **Rider（新）** | `uid_days_since_first` | 乘客從首次叫車到當筆的天數 | 取代 `uid_hash_te` |
| Location | `start_h3_te`, `start_town_te` | 同 A | 不變 |
| Demand | `demand_town_hour` | 同 A | 不變 |
| **Destination（新）** | `end_town` | 下車行政區，native categorical | 全新 |

**司機三統計設計：**

- 計算時排除春節（2026-02-14 ~ 2026-02-22），避免節慶異常污染常態偏差
- 司機資料少於 10 筆者，std 填全域中位數（統計不穩定）
- avg/median 以 Laplace smoothing 向全域均值拉（prior weight m=30）
- Train 自身用 5-fold OOF；valid/test 用全 train 統計

**乘客特徵改為活躍度：**

`uid_hash_te` 每位乘客平均只有 4 筆，target encoding 大概率在記憶雜訊。permutation importance 驗證：uid 相關特徵合計僅 1.8%。改用純計數特徵，不接觸 label，更穩健。

## 5. Model Method

超參數借用 Pipeline C 的 Optuna 30-trial 最佳結果：

```
learning_rate=0.0499, num_leaves=120, min_child_samples=262,
feature_fraction=0.605, bagging_fraction=0.945, bagging_freq=3,
lambda_l1=3.46e-4, lambda_l2=0.0677, rounds=1000（保守值）
```

新增 **Permutation Importance**（validation set，每特徵打亂 5 次）：

| 排名 | 特徵 | MAE 上升 | 佔比 |
|---|---|---|---|
| 1 | `log_eta` | +46.65s | 66.5% |
| 2 | `hour_cos` | +4.95s | 7.1% |
| 3 | `driver_avg_logratio` | +3.77s | 5.4% |
| 4 | `hour_sin` | +3.40s | 4.8% |
| 5 | `end_town` | +2.11s | 3.0% |
| 6 | `start_h3_te` | +1.82s | 2.6% |
| 7 | `driver_median_logratio` | +1.72s | 2.4% |
| 8 | `driver_std_logratio` | +1.22s | 1.7% |
| 14 | `is_special_day` | +0.00s | 0.0% |

## 6. Results

| 模型 | MAE↓ | RMSE↓ | MeanErr | Within60↑ | Within120↑ | P90↓ | Late>60↓ | MAE 改善 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 88.1s | 128.0s | -39.5s | 49.6% | 76.0% | 196s | 14.7% | — |
| **LightGBM** | **76.4s** | **110.6s** | -18.9s | 54.3% | 81.3% | 166s | 18.6% | **-13.3%** |
| **Quantile q50** | **76.0s** | **109.9s** | -14.3s | 54.3% | 81.5% | 165s | 20.1% | **-13.7%** |

---

---

# Pipeline C — final 自動搜尋版

## Overall Pipeline

```
Data Input → Data Cleaning → Time-based Split → Feature Superset 建構（18 blocks）
→ Greedy Forward Feature Search → Optuna HP Tuning（30 trials）
→ Final Retrain（train + valid）→ Evaluation
```

## 1. Data Input

同 Pipeline A/B，額外使用目的地座標：

- `end_lat`, `end_lng`：缺失 2.8% → 以上車座標填補（haversine 距離輸出 0 作為訊號）
- `end_county`, `end_town`：缺失 2.8% → 以上車行政區填補

乘客特徵（`uid_hash`）完全排除：每人平均 4 趟，noise > signal。

## 2. Data Cleaning

同 Pipeline A/B，完全一致的清洗口徑（1,462,517 筆）。

**Target 定義（加法殘差）：**

```
target = time_accept_to_arrive − driver_eta
corrected_eta = driver_eta + pred
```

## 3. Time-based Split

Pipeline C 採用月份制切分（test set 較 A/B 大）：

| Split | 月份 | 筆數 |
|---|---|---|
| Train | Jan ~ Mar 2026 | 942,304 |
| Valid | Apr 2026 | 314,220 |
| Test | May 2026 | 205,795 |

## 4. Feature Engineering

**18 個 FeatureBlock 架構**，每個 block 有 `fit(train)` 和 `transform()` 方法，保證 stateful 統計只在 train 上計算。

**Greedy Forward Feature Search：**

| 步驟 | 加入 block | Val MAE | 改善 |
|---|---|---|---|
| 起點 | `base_eta` | 85.47s | — |
| 1 | `time_basic` | 80.64s | -4.83s |
| 2 | `time_cyclic` | 80.60s | -3.32s |
| 3 | `driver_bias` | 78.17s | -3.28s |
| 4 | `time_flags` | 78.17s | -1.64s |
| 5 | `region_cat` | 77.28s | -0.86s |
| 6 | `geo_raw` | 77.06s | -0.69s |
| 7 | `geo_distance` | 77.06s | -0.21s |
| 停止 | 其餘 block 改善 < 0.2s | | |

## 5. Model Method

**Optuna 30 trials：**

最佳超參數：`lr=0.0499, num_leaves=120, min_child_samples=262, feature_fraction=0.605, bagging_fraction=0.945`

最終在 train+valid 合併後訓練，在全 5 月 test set 評估。

## 6. Results

| 模型 | MAE↓ | RMSE↓ | MeanErr | Within60↑ | Within120↑ | P90↓ | MAE 改善 |
|---|---|---|---|---|---|---|---|
| Baseline | 88.5s | 127.5s | -39.4s | 49.4% | 75.7% | 197s | — |
| **final LightGBM** | **75.8s** | **109.3s** | -17.0s | 54.5% | 81.6% | 165s | **-14.3%** |

---

---

# Pipeline D — DoDo（CatBoost + 天氣）

## Overall Pipeline

```
Data Input → Data Cleaning → Time-based Split → Feature Engineering
→ CatBoost Residual Model → Post Calibration → Prediction Output → Evaluation
```

## 1. Data Input

主要使用 `trip_stats_eta.parquet`，並額外引入**外部天氣資料**。

主要欄位（與 A/B/C 相同）：

- `driver_eta`：API 原始預估司機抵達乘客位置的時間
- `time_accept_to_arrive`：司機實際抵達乘客位置的時間，ground truth 來源
- `request_time`：叫車時間，用來建立 time features 與時間切分
- `start_lat`, `start_lng`, `start_county`, `start_town`：上車地點資訊
- `did_hash`：司機 ID，用來學習不同司機的固定偏差
- `uid_hash`：乘客 ID，目前有保留作為 feature，但效果小於 driver ID
- `end_*`：目的地相關欄位，作為 destination features

**外部輔助資料（Pipeline D 特有）：**

- 天氣資料：從交通部中央氣象署（CWA）公開資料集取得全台灣各觀測站歷史天氣數據
- 以 trip 的起點座標對應最近觀測站，取該小時的最新天氣記錄
- 透過 trip-level feature file 合併，match rate 約 99.9%
- 天氣 cache 獨立建立，不直接修改原始 trip dataset

## 2. Data Cleaning

主要處理：

- 移除缺少上車位置（start latitude / longitude / county / town）的資料
- 移除 `driver_eta <= 0` 的資料
- 移除 `time_accept_to_arrive <= 0` 的資料
- 移除極端行程：`driver_eta >= 1050s` 或 `time_accept_to_arrive >= 1380s`
- 不直接修改原始 parquet，只在 training pipeline 中過濾

Leakage control：

- `time_accept_to_arrive` 只用來建立 target，不作為 model input
- `time_start_to_finish` 不使用（乘客叫車時不會知道此資訊）
- 歷史統計與 rolling features 只使用過去資料，並加入時間 lag

## 3. Time-based Split

| Split | 日期範圍 | 用途 |
|---|---|---|
| Train | 1 月至 3 月 | 訓練模型 |
| Valid | 4 月 | 選擇模型、特徵與 calibration 策略 |
| Test | 5 月 | 最終評估 |

## 4. Feature Engineering

主要 feature 類型：

| 類型 | 特徵 | 說明 |
|---|---|---|
| ETA | `driver_eta`, `log_driver_eta`, `sqrt_driver_eta`, `driver_eta_sq`, `eta_bucket`, `eta_bin` | ETA 多種變換與分段 |
| Time | `hour`, `minute`, `weekday`, `day_of_year`, `rush_hour`, `night`, `weekend`, `hour_sin/cos` | 時間週期與特殊時段 |
| Location | `start_county`, `start_town`, `start_grid` | 縣市、行政區、100m×100m 方格 |
| Driver | `did_hash`, `did_hour`, `did_start_town`, `did_eta_bin`, `did_start_town_eta_bin` | 司機個人偏差與多維交互 |
| Demand | 近期叫車量（10/30/60/120/240 分鐘視窗） | 供需鬆緊代理 |
| Weather | `weather_temp_c`, `weather_rain_mm`, `weather_has_rain`, `weather_rain_log`, `rain_x_eta` | 天氣影響（CWA 資料） |
| Destination | `end_county`, `end_town`, `end_lat`, `end_lng`、OD distance、OD region | 目的地與起終點組合 |

**Region Grid（100m×100m 方格）：**

使用 square grid（非 H3），grid size = 100m×100m（`grid_type=square_m`, `square_size_m=100`）。同時使用 county、town、grid 三種尺度。

**Region-Time Features：**

建立 `start_town_hour`, `start_county_hour`, `start_grid_hour`，捕捉「某地區在某時段」的交通與叫車模式。HourType 分類參考 LINEGO 第一題並擴充。

**ETA Bucket：**

將 `driver_eta` 分段（短/中/長 ETA）。同時建立 `eta_bin_hour` 學「ETA 區間 × 時段」的交互效果。

**Driver Interaction：**

建立 `did_hour`, `did_start_town`, `did_eta_bin`, `did_start_town_eta_bin`，描述同一司機在不同時間、地區、ETA 區間下的偏差。

**Rolling History：**

計算 driver、region、ETA bucket 的歷史平均誤差與 MAE，加入 30 分鐘 label lag，避免使用當下或未來才知道的 ground truth。

**Demand Proxy：**

使用 10/30/60/120/240 分鐘滑動視窗，計算不同 key（driver、driver-hour、driver-start-county、driver-start-town）下的 request count 與平均 ETA。

**Weather Features（Pipeline D 特有）：**

- 來源：中央氣象署（CWA）公開資料集
- 特徵：`weather_temp_c`, `weather_rain_mm`, `weather_has_rain`, `weather_rain_log`
- 交互特徵：`rain_x_eta`（雨量 × ETA 長短同時影響誤差）
- Missing flag：避免模型把缺失天氣資料誤解為沒有下雨

## 5. Model Method

**主模型：CatBoost Regression**

選擇 CatBoost 的原因：能原生處理大量 categorical features（driver ID、行政區、grid、ETA bucket），無需手動編碼。

模型預測 API 的**加法殘差**：

```
target = time_accept_to_arrive − driver_eta
corrected_eta = driver_eta + pred
```

## 6. Post Calibration

模型預測後進行 residual calibration，根據 validation set 觀察不同群組的系統性偏差再做小幅修正：

1. 先依照 driver ID 修正
2. 再依照 ETA bucket 修正

實作方式（不造成 test leakage）：

- 使用 validation split 前段 fit calibration table
- 使用 validation split 後段選擇最佳 calibration strategy
- 最後才套用到 test set

## 7. Auxiliary Experiment

Two-stage 實驗（非主方法）：

- Stage 1：將 API error 分成 over-estimate、normal、under-estimate 三類
- Stage 2：各類別使用 specialist model 修正 residual
- 最後與主模型做 weighted blend（25% two-stage + 75% 主模型）可小幅改善

## 8. Results

| Metric | Baseline | Linear | Lasso (L1) | Ridge (L2) | **CatBoost (Ours)** |
|---|---|---|---|---|---|
| **MAE (s)** | 88.93 | 82.02 | 82.26 | 82.04 | **75.17** |
| **RMSE (s)** | 129.12 | 116.43 | 116.74 | 116.46 | **108.58** |
| **Within 60s** | 49.35% | 49.46% | 49.30% | 49.47% | **54.65%** |
| **Within 120s** | 75.68% | 79.50% | 79.43% | 79.48% | **81.95%** |
| **Mean Error (s)** | 39.20 | 6.33 | 5.45 | 6.37 | 14.01 |
| **MAE 改善** | — | -7.77% | -7.50% | -7.75% | **-15.47%** |
| **RMSE 改善** | — | -9.83% | -9.58% | -9.81% | **-15.91%** |

Linear/Lasso/Ridge 改善約 7.5~8%，顯示線性模型能捕捉系統性偏差但有天花板。CatBoost 達到 -15.47%，非線性交互特徵的效果顯著。

---

---

# 跨方法比較

## 主結果對照表

> **Test set 說明：**
> - Pipeline A/B：2026-05-07 ~ 05-20，144,605 筆，Baseline MAE = 88.1s
> - Pipeline C：整個 2026 年 5 月，205,795 筆，Baseline MAE = 88.5s
> - Pipeline D：整個 2026 年 5 月（月份制），Baseline MAE = 88.93s
>
> 因 test set 邊界略有差異，以**改善幅度（%）**作為跨方法主要比較依據。

| 模型 | MAE | RMSE | Within60 | Within120 | MeanErr | MAE 改善 | RMSE 改善 |
|---|---|---|---|---|---|---|---|
| Baseline | ~88.5s | ~128s | ~49.5% | ~75.8% | ~39s | — | — |
| **D: Linear** | 82.0s | 116.4s | 49.5% | 79.5% | 6.3s | -7.8% | -9.1% |
| **D: Lasso** | 82.3s | 116.7s | 49.3% | 79.4% | 5.5s | -7.5% | -8.9% |
| **D: Ridge** | 82.0s | 116.5s | 49.5% | 79.5% | 6.4s | -7.8% | -9.0% |
| A: src LightGBM | 76.6s | 111.1s | 54.2% | 81.3% | -20.5s | -13.2% | -13.2% |
| A: src q50 | 76.1s | 110.2s | 54.2% | 81.7% | -15.8s | -13.6% | -13.9% |
| B: mix LightGBM | 76.4s | 110.6s | 54.3% | 81.3% | -18.9s | -13.3% | -13.5% |
| B: mix q50 | 76.0s | 109.9s | 54.3% | 81.5% | -14.3s | -13.7% | -14.1% |
| C: final LightGBM | 75.8s | 109.3s | 54.5% | 81.6% | -17.0s | -14.3% | -14.3% |
| **D: CatBoost** | **75.2s** | **108.6s** | **54.7%** | **82.0%** | 14.0s | **-15.5%** | **-15.9%** |

## 改善幅度階梯

```
Baseline                   → MAE ~88.5s  (0%)
├── Linear / Lasso / Ridge → MAE ~82.0s  (約 -7.5~8%)   線性模型上限
├── A/B: LightGBM          → MAE 76.0~76.6s (-13~14%)
├── C: final LightGBM      → MAE 75.8s   (-14.3%)        自動搜尋 + 調參
└── D: CatBoost            → MAE 75.2s   (-15.5%)        豐富特徵 + 天氣 + 校正
```

## 特徵設計比較

| 面向 | A — src | B — mix | C — final | D — DoDo |
|---|---|---|---|---|
| 特徵數 | 11 | 15 | 28 | 50+ |
| 司機特徵 | 1 個 TE | 3 個統計量（OOF） | 3 個統計量（OOF） | TE + 多維交互 |
| 乘客特徵 | uid TE | 次數 + 天數 | 不使用 | uid TE |
| 地理精細度 | H3 ~170m | H3 ~170m | 原始座標 + 幾何 | 100m×100m Grid |
| 目的地 | 無 | end_town | end_town/county/座標 | end_town/county + OD region |
| 天氣 | 無 | 無 | 無 | 有（CWA，match rate 99.9%） |
| Rolling history | 無 | 無 | 無 | 有（30 min lag） |
| 後處理校正 | 無 | 無 | 無 | 有（driver + ETA bucket） |
| 模型 | LightGBM | LightGBM | LightGBM | CatBoost |
| 調參 | 無 | 借用 C | Optuna 30 trials | — |

## Ablation Summary

以下整理各設計決策的邊際貢獻，資料來自已有的實驗對比，不需額外重跑：

| 設計改動 | 基準 MAE | 改後 MAE | 差距 | 資料來源 |
|---|---|---|---|---|
| 安裝 H3（無 → 有） | 76.9s（A 無 H3） | 76.6s（A 有 H3） | **-0.3s** | H3 安裝前後直接對比 |
| 司機：TE → 三統計量 | 76.6s（A） | 76.4s（B） | -0.2s | A vs B 對比（含其他改動） |
| 新增 end_town | 76.6s（A） | 76.4s（B） | -0.2s | permutation：3.0% 貢獻 |
| 移除 uid_hash_te | ≈76.6s | ≈76.4s | 無明顯退步 | permutation：uid 僅 1.8% |
| 預設參數 → Optuna 30 trials | ~76.6s（A） | 75.8s（C） | **-0.8s** | A vs C 對比（含特徵差異） |
| Greedy search：base → 7 blocks | 85.5s | 77.1s（val） | **-8.4s（val）** | C 的 forward trace |
| 加入天氣 + rolling + 校正（D） | 75.8s（C） | 75.2s（D） | **-0.6s** | C vs D 對比（多項差異混合） |

> 注意：A vs B 的差距混合了司機特徵、乘客特徵、end_town 三項改動，無法單獨歸因。天氣貢獻亦與 rolling history、後處理校正等混合，需控制實驗才能單獨量化。

---

---

# 亮點與關鍵觀察

## 觀察一：線性模型的天花板

Linear / Lasso / Ridge 三者 MAE 都在 82s 左右（改善約 7.5%），彼此差距不到 0.3s。

1. **線性模型能捕捉 API 的系統性偏差**（88s → 82s），但有明顯天花板
2. **正則化（L1/L2）對本任務幾乎沒有差別**——資料量大（140 萬筆），不存在過擬合問題
3. **非線性交互效果不可少**：LightGBM / CatBoost 額外再降 7s，純粹來自非線性和 feature interaction

## 觀察二：log_eta 主導 gain importance 的數學原因

在 Pipeline A/B（log-ratio target）中，`log_eta` 的 gain importance 高達 **50~55%**。原因是 target 公式的數學結構：

```
target = log(actual) − log(driver_eta) = log(actual) − log_eta
```

`log_eta` 直接嵌入 target 公式，模型必然重度依賴它。Pipeline C 採用加法殘差，`driver_eta` 的 gain 降到約 12%，分布更均勻。

**結論**：gain importance 在 log-ratio target 下對 `log_eta` 嚴重高估，permutation importance 才是公允衡量。

## 觀察三：Permutation vs Gain — 兩種重要度的差距

| 特徵 | Gain % | Permutation % | 解讀 |
|---|---|---|---|
| `log_eta` | 48.5% | **66.5%** | Permutation 更高：真實最重要 |
| `start_h3_te` | 5.8% | 2.6% | Gain 高估：高基數被選到多次但增益低 |
| `end_town` | 8.2% | 3.0% | Gain 略高估，但 permutation 確認有效 |
| `is_special_day` | — | **0.0%** | 兩種都說無用 |

## 觀察四：uid_hash_te 幾乎是雜訊

Pipeline A 的 `uid_hash_te`：split 11.1%、gain 8.6%，看似重要。Pipeline B 改用活躍度特徵後，permutation 合計僅 **1.8%**。每位乘客平均只有 4 筆資料，target encoding 在記憶雜訊。

**結論**：可安心移除乘客 TE，不影響整體 MAE。

## 觀察五：H3 缺失是關鍵瓶頸

H3 套件未安裝時，`start_h3_te` 全部輸出 "unknown"，permutation importance 為 **0.0%**。安裝後：

- Pipeline A MAE：76.9s → **76.6s**（-0.3s）
- Pipeline B MAE：76.8s → **76.4s**（-0.4s）
- H3 permutation importance 提升至 **2.6%**（排名第 6）

H3 解析度 res-9（~170m）能捕捉行政區（~數公里）捕捉不到的局部熱點，如特定路口、停車難度高的地點。

## 觀察六：API 系統性低估的不對稱性

原始 API 誤差：Mean error = -39.5s（實際比 API 慢），晚到 >60s 佔 14.7%，早到 >60s 佔 35.6%。

校正後（Pipeline A q50）：Mean error = -15.8s，晚到 19.7%，早到 26.1%。

晚到比例從 14.7% 升至 19.7%，原因是大量的早到被修正成接近準確，邊界附近的晚到因此更顯眼。整體 MAE/RMSE 顯著改善，且 mean error 從 -39.5s 縮至 -15.8s，校正有效。

## 觀察七：q50 vs L2 — 建議以 q50 作為主要點估計

| | L2（MSE） | q50（MAE） |
|---|---|---|
| 損失函數 | 均方誤差 | 絕對誤差 |
| 預測目標 | 條件平均值 | 條件中位數 |
| 對離群值 | 受大誤差拉偏 | 穩健 |
| MAE（本資料集） | 76.6s（A）/ 76.4s（B） | 76.1s（A）/ 76.0s（B） |

在有明顯長尾的 ETA 誤差分布中，q50 在 MAE 和 RMSE 上均優於 L2，建議以 q50 作為主要點估計，q90 作為「保守模式」的可選輸出。

## 觀察八：天氣與外部資訊的邊際增益

Pipeline D（有天氣）vs Pipeline C（無天氣）：MAE 差距約 0.6s（75.2 vs 75.8s）。但兩者同時包含 rolling history、更細的 driver 交互、後處理校正等差異，天氣本身的單獨貢獻無法直接量化。

天氣資料的主要價值可能集中在雨天等極端場景的誤差縮小，而非整體均值。CWA 開放資料 match rate 99.9%，技術上整合成本低。

---

*本文件為 draft 版本，後續組員可依各節內容進行修改、補充數字或調整措辭。*
