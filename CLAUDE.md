# CLAUDE.md — ETA 抵達時間校正專案

> 這份文件是給 **Claude Code** 在本地端閱讀並協助執行用的專案規格與脈絡。
> 它記錄了專案目標、資料、已驗證的發現、方法論決策、程式架構、每階段的起手 code、評估框架,以及所有重要的陷阱與注意事項。
> 開發者環境:**Mac M1 / 16GB RAM**,本地執行,使用 Polars + LightGBM。

---

## 0. 一句話摘要

這是一個 **ETA(預估司機抵達時間)校正** 專案。叫車平台的 routing API 給的「司機抵達上車點時間」存在**系統性低估**(平均晚到 +76 秒)。我們的目標是:**站在 API 的估計值上,用機器學習學它「錯多少」,把校正後的 ETA 變得更準,特別是壓低長尾的大誤差。**

核心策略是 **殘差學習(residual learning)**:不直接預測抵達時間,而是預測 API 的誤差,再加回去。

---

## 1. 專案背景與問題定義

### 1.1 課程脈絡
- 這是一門課程的期末小組作業(LINE GO 提供題目),五人小組。
- 我們選的是「**題目二:預估司機抵達時間校正**」。
- 痛點(題目原文意旨):API 估計「司機接到乘客的抵達時間」常不準,導致 (1) 派遣可能選錯司機 (2) 用戶看到的「預期抵達」與「真實抵達」落差大(例:顯示 3 分鐘、實際 6 分鐘)。

### 1.2 為什麼用殘差學習(關鍵觀念,務必遵守)
- API 已經給了一個估計值 `driver_eta`,它內含了 routing 引擎用多年路網資料算出的結果。我們**不該、也無法**從零重建一個 routing 引擎。
- 正確切入點:
  ```
  真實抵達時間 = API估計值(driver_eta) + 殘差(residual)
  ```
  模型只需學那個 **residual**(系統性偏差),任務小得多、資料需求也小。
- 這跟 Uber DeepETA、DiDi WDR 的工業界做法一致。

---

## 2. 資料規格(Data Schema)

- **檔名**:`trip_stats_eta`(parquet)
- **規模**:1.14 GB,**1,546,557 筆**,16 欄
- **時間範圍**:2026/01 ~ 2026/05(約 120 天)

| 欄位 | 型態 | 說明 | 缺失/備註 |
|---|---|---|---|
| `trip_id` | STRING | 行程唯一識別碼 | 0 缺失 |
| `uid_hash` | STRING | 乘客匿名碼(SHA256) | 0 缺失;**374,059** 個不同乘客(平均每人 ~4 趟,個人化價值低) |
| `did_hash` | STRING | 司機匿名碼(SHA256) | 0 缺失;**14,674** 個不同司機(平均每司機 **~105 趟**,**強特徵**) |
| `start_lat` | FLOAT64 | 上車緯度 | 0 缺失 |
| `start_lng` | FLOAT64 | 上車經度 | 0 缺失 |
| `start_county` | STRING | 上車縣市 | 僅 17 筆缺失 → 直接丟棄該 17 筆 |
| `start_town` | STRING | 上車行政區 | 僅 17 筆缺失 |
| `end_lat` | FLOAT64 | 下車緯度 | ~43,682 缺失;**本專案不太需要** |
| `end_lng` | FLOAT64 | 下車經度 | ~43,682 缺失;不太需要 |
| `end_county` | STRING | 下車縣市 | ~43,727 缺失;不太需要 |
| `end_town` | STRING | 下車行政區 | ~43,727 缺失;不太需要 |
| `end_address` | STRING | 下車完整地址 | ~1,989 缺失;不太需要 |
| `request_time` | TIMESTAMP | 乘客發起叫車時間(**Unix 秒**) | 0 缺失;**時區為 UTC,需 +8 小時轉台灣當地** |
| `driver_eta` | INT64 | **API 預估**司機抵達上車點時間(秒) | target 來源之一 |
| `time_accept_to_arrive` | INT64 | **實際**司機抵達上車點時間(秒) | **target 來源**(真實值) |
| `time_start_to_finish` | INT64 | 行程開始到結束時間(秒) | 本專案不太需要 |

### 2.1 資料品質結論(已由組員驗證)
- 建模需要的核心欄位(上車座標、時間、`driver_eta`、`time_accept_to_arrive`)**全部完整**。
- 有缺失的都是 `end_*`(下車欄位),而那些對「接駁 ETA」關聯不大,可忽略。
- `start_lat`/`start_lng` distinct 約 88 萬(57%)→ 座標幾乎筆筆不同,**必須用 H3 歸併成區塊才有 pattern 可學**。

---

## 3. 已驗證的關鍵發現(EDA 結論,直接拿來指導建模)

### 3.1 API 系統性低估(專案核心前提,已成立)
- 全量 `mean_error = +76 秒`、`median_error = +29 秒`(`time_accept_to_arrive - driver_eta`)。
- 晚到 >60 秒:**37.7%**;早到 >60 秒:**15.1%**。晚到是早到的 2.5 倍。
- → 偏差是**系統性、單向(低估)**的,所以可學。模型整體往上修,並對特定情境修更多。

### 3.2 誤差集中在長尾(主戰場)
- 全量 `MAE = 126`、`RMSE = 265`。**RMSE 遠大於 MAE = 有一群離譜大誤差。**
- 百分位數加速爬升:`P50=65, P80=155, P90=246, P95=418`(秒)。最後 5~10% 的行程誤差暴衝。
- **價值最大的地方是壓低長尾**(P90/P95),而非把已準的中位數再擠幾秒。

### 3.3 誤差大小與「估計值長短」無關
- 樣本觀察:估最久(475s)的反而最準(+3s);估最短(151s)的誤差最大(+265s)。
- → 誤差不由路程長短決定,而由**上車點情境**決定(難找、難停、難接)。這是「上車點難找度」假設的依據。

### 3.4 月份穩定,信號在更細尺度
- 五個月 MAE 全在 124~128、mean_error 全在 74~77 → **「月份」幾乎不帶資訊**,別做月特徵。
- 但**特定日會爆糟**:`2/16~2/19`(2026 農曆春節,2/17 除夕)出現「**低叫車量 + 高 MAE**」。
- → 真正有用的時間信號在 **星期/小時/是否假日/是否特殊日(春節、連假、颱風)**,以及**供需鬆緊(叫車量)**。

### 3.5 司機 vs 乘客特徵價值
- `did_hash` 平均 ~105 趟 → target encoding 統計穩定,**強特徵,優先做**。
- `uid_hash` 平均 ~4 趟 → 雜訊大,target encoding 需**很重的平滑**(往全域平均拉),價值低。

---

## 4. 目前進度(已完成 / 待辦)

```
[x] 1. 資料品質檢查(缺失、distinct、分布)
[x] 2. Baseline 計算(raw driver_eta,在 test set 上 MAE 123.6 / RMSE 246.6)
[x] 3. Linear / Ridge 回歸(殘差加法版):
       - Linear: MAE 108.0(-12.6%)、RMSE 173.8(-29.5%)、mean_error +77→+1.1
       - Ridge 幾乎相同 → 目前模型「太簡單」而非過擬合 → 該上 LightGBM
       - 副作用:晚到↓(38%→26%,好)但早到↑(14%→31%);這是線性模型「平均主義」矯枉過正,可接受(早到傷害遠小於晚到),之後用分位數模型主動控制
[ ] 4. H3 + 時間特徵 + OOF target encoding
[ ] 5. LightGBM 主模型(吃非線性與交互作用,目標再壓低 P90/P95)
[ ] 6. 分位數 LightGBM(q=0.1/0.5/0.9,給「5–7 分鐘」區間,主動偏向「寧早勿晚」)
[ ] 7. 評估視覺化 + 切片分析(town / 時段 / 特殊日 / 天氣)
[ ] 8. (加分)天氣特徵(中央氣象署,縣市+時間 join)
```

### 已知須與組員確認的兩件事
1. **資料切分必須「照時間連續」切**:train 的所有日期 < valid 的所有日期 < test。不能隨機挑天。
   - 目前 test = `2026-05-07 ~ 2026-05-20`(最後 14 天,152,669 筆)。
2. **天數對帳**:組員寫 98/28/14=140 天,但資料約 120 天,需確認實際天數。

---

## 5. 方法論決策(建模規範)

### 5.1 Target 定義(建議用 log-ratio)
目前組員用的是**加法殘差**:`target = time_accept_to_arrive - driver_eta`,還原 `corrected_eta = max(0, driver_eta + pred)`。這版已經 work(RMSE -29.5%)。

**主模型建議改用 log-ratio**(對長尾更穩):
```
target = log(time_accept_to_arrive / driver_eta)
還原:corrected_eta = driver_eta * exp(pred)
```
理由:加法殘差會被長程主導(估 10 分鐘的誤差絕對值通常比估 2 分鐘大);log-ratio 對乘法性誤差對稱、更穩。比值都 ≥1,log 後集中在 0 以上,適合迴歸。

> **執行建議**:兩種 target 都跑、在 test set 上比 P90/RMSE,擇優。文件預設以 log-ratio 為主。

### 5.2 資料切分
- **時間連續切**,7:2:1(train:valid:test)。
- target encoding 等任何「用到 label 的統計量」**只能用 train 那段時間算**,嚴禁洩漏。
- 注意:同一個 `did_hash` 會同時出現在 train/test(司機重複),這是**真實上線情境**(面對見過的司機),OK;但 TE 統計量仍只能用 train 算。

### 5.3 清洗紅線
- 丟棄:`driver_eta <= 0`、`time_accept_to_arrive <= 0`、`start_county` 為 null 的 17 筆。
- 長尾上限:先看 `time_accept_to_arrive` 的 P99.9 再定上限(接駁 >30 分鐘的本來就少且性質不同,可砍)。**不要硬套固定值,看分布決定。**

---

## 6. 程式架構(四階段 pipeline)

每階段輸出落地成 parquet,任一階段壞掉可單獨重跑,不必每次從 1.14GB 重來。

```
src/
├── config.py          # 路徑、欄位、切分日期、參數常數
├── load_clean.py      # 讀檔 + 清洗 + 算 target + 時間切分 → trips_clean.parquet
├── features.py        # H3 + 時間特徵 + OOF target encoding → feats.parquet
├── weather.py         # (加分)接氣象資料,縣市+時間 join
├── train.py           # baseline + LightGBM + 分位數模型
└── evaluate.py        # 指標 + 校準圖 + 切片分析 + 視覺化

data/
├── raw/trip_stats_eta.parquet     # 原始(共用,勿改)
├── trips_clean.parquet            # load_clean 產物
└── feats.parquet                  # features 產物

outputs/
├── metrics/                       # 各模型指標 csv/json
├── figures/                       # 圖表
└── models/                        # 訓練好的模型
```

**接口契約(schema contract)**:各階段 parquet 欄位需白紙黑字定義,改欄位先通知全組。清洗這步由「第一人」跑完,把 `trips_clean.parquet` 丟共用雲端,其他人下載同一份,避免版本分歧。

---

## 7. 各階段可執行起手 code

> 以下為起手骨架,Claude Code 可據此擴充。Python 3.11,Polars 為主。

### 7.1 `config.py`
```python
from pathlib import Path

DATA_RAW   = Path("data/raw/trip_stats_eta.parquet")
DATA_CLEAN = Path("data/trips_clean.parquet")
DATA_FEATS = Path("data/feats.parquet")
OUT        = Path("outputs")

TZ_OFFSET_HOURS = 8           # request_time 是 UTC,+8 轉台灣
H3_RES_FINE   = 9             # ~170m,捕捉「最後一哩」
H3_RES_COARSE = 8             # 兜底

# 時間切分(連續切!待確認實際天數後填正確日期)
# train < valid < test;test 固定 2026-05-07 ~ 2026-05-20
SPLIT = {
    "valid_start": "2026-04-23",   # 範例,依實際天數調整
    "test_start":  "2026-05-07",
}

# TE 平滑強度(司機資料多用小 m,乘客資料少用大 m)
TE_SMOOTH = {"start_h3": 20, "start_town": 50, "did_hash": 30, "uid_hash": 200}

LGB_PARAMS = dict(
    n_estimators=800, learning_rate=0.03, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
)
```

### 7.2 `load_clean.py`
```python
import polars as pl
from config import DATA_RAW, DATA_CLEAN, TZ_OFFSET_HOURS

def main():
    lf = pl.scan_parquet(DATA_RAW)  # lazy,省記憶體

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

    # 長尾上限:看 P99.9 再砍(這裡示範,實際印出分布後決定)
    hi = df["time_accept_to_arrive"].quantile(0.999)
    df = df.filter(
        (pl.col("time_accept_to_arrive") < hi) & (pl.col("driver_eta") < hi)
    )

    # target:log-ratio(主)+ 加法殘差(備)
    df = df.with_columns(
        (pl.col("time_accept_to_arrive") / pl.col("driver_eta")).log().alias("target_logratio"),
        (pl.col("time_accept_to_arrive") - pl.col("driver_eta")).alias("target_additive"),
        pl.col("local_dt").dt.date().alias("date"),
    )

    df.write_parquet(DATA_CLEAN)
    print(f"cleaned rows: {df.height}, date range: {df['date'].min()} ~ {df['date'].max()}")

if __name__ == "__main__":
    main()
```

### 7.3 `features.py`(含無洩漏 OOF target encoding)
```python
import h3
import numpy as np
import polars as pl
from config import DATA_CLEAN, DATA_FEATS, H3_RES_FINE, H3_RES_COARSE, TE_SMOOTH, SPLIT

def add_h3(df: pl.DataFrame) -> pl.DataFrame:
    lat = df["start_lat"].to_numpy(); lng = df["start_lng"].to_numpy()
    fine   = [h3.latlng_to_cell(a, b, H3_RES_FINE)   for a, b in zip(lat, lng)]
    coarse = [h3.latlng_to_cell(a, b, H3_RES_COARSE) for a, b in zip(lat, lng)]
    return df.with_columns(
        pl.Series("start_h3",  fine),
        pl.Series("start_h3c", coarse),
    )

def add_time(df: pl.DataFrame) -> pl.DataFrame:
    h = pl.col("local_dt").dt.hour()
    return df.with_columns(
        (2*np.pi*h/24).sin().alias("hour_sin"),
        (2*np.pi*h/24).cos().alias("hour_cos"),
        pl.col("local_dt").dt.weekday().alias("dow"),
        (pl.col("local_dt").dt.weekday() >= 6).alias("is_weekend"),
        pl.col("driver_eta").log().alias("log_eta"),  # 原始估值當特徵
        # TODO: is_holiday / is_special_day 需 join 台灣假日表(見 §8)
    )

def smoothed_te(train: pl.DataFrame, key: str, target: str, m: int):
    g = train[target].mean()
    agg = (train.group_by(key)
                 .agg(pl.col(target).mean().alias("m"), pl.col(target).count().alias("n"))
                 .with_columns(((pl.col("n")*pl.col("m") + m*g)/(pl.col("n")+m)).alias(f"{key}_te"))
                 .select(key, f"{key}_te"))
    return agg, g

def main(target_col: str = "target_logratio"):
    df = pl.read_parquet(DATA_CLEAN)
    df = add_h3(df)
    df = add_time(df)

    # 切分(連續)
    is_test  = pl.col("date") >= pl.lit(SPLIT["test_start"]).str.to_date()
    is_valid = (pl.col("date") >= pl.lit(SPLIT["valid_start"]).str.to_date()) & ~is_test
    df = df.with_columns(
        pl.when(is_test).then(pl.lit("test"))
          .when(is_valid).then(pl.lit("valid"))
          .otherwise(pl.lit("train")).alias("split")
    )

    train = df.filter(pl.col("split") == "train")

    # === OOF target encoding(嚴防洩漏)===
    # 1) valid/test:直接用「全 train」算的 TE 套上去,未見過的 key 填全域平均 g
    # 2) train 自身:用 5-fold OOF(用其他 fold 算、套當前 fold),否則模型偷看自己答案
    for key, m in TE_SMOOTH.items():
        agg, g = smoothed_te(train, key, target_col, m)
        df = df.join(agg, on=key, how="left").with_columns(pl.col(f"{key}_te").fill_null(g))
        # TODO(Claude Code): 將 train 段的 {key}_te 改為 5-fold OOF 重算,避免 train 洩漏

    df.write_parquet(DATA_FEATS)
    print("features written:", df.columns)

if __name__ == "__main__":
    main()
```

> **重點提醒給 Claude Code**:上面 TE 對 valid/test 是正確的(用全 train 算),但 **train 段需改成 5-fold OOF** 才嚴謹(目前是偷懶版,會讓 train 洩漏)。請實作 OOF 版本:把 train 切 5 折,每折的 TE 用「其他 4 折」算。

### 7.4 `train.py`
```python
import lightgbm as lgb
import numpy as np
import polars as pl
from config import DATA_FEATS, LGB_PARAMS

FEATURES = ["log_eta", "hour_sin", "hour_cos", "dow", "is_weekend",
            "start_h3_te", "start_town_te", "did_hash_te", "uid_hash_te"]  # 視 features 產物調整

def to_xy(df, target_col):
    X = df.select(FEATURES).to_pandas()
    y = df[target_col].to_numpy()
    return X, y

def main(target_col="target_logratio"):
    df = pl.read_parquet(DATA_FEATS)
    tr = df.filter(pl.col("split") == "train")
    va = df.filter(pl.col("split") == "valid")

    Xtr, ytr = to_xy(tr, target_col)
    Xva, yva = to_xy(va, target_col)

    # 點估計模型
    model = lgb.LGBMRegressor(objective="l2", **LGB_PARAMS)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    # 分位數模型(給區間,主動偏「寧早勿晚」)
    qmodels = {}
    for q in [0.1, 0.5, 0.9]:
        qmodels[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS)
        qmodels[q].fit(Xtr, ytr, eval_set=[(Xva, yva)],
                       callbacks=[lgb.early_stopping(50)])

    # TODO: 存模型、印 feature_importances_(報告要用)
    return model, qmodels

if __name__ == "__main__":
    main()
```

> 還原預測:log-ratio 版 `corrected_eta = driver_eta * exp(pred)`;加法版 `max(0, driver_eta + pred)`。分位數可能交叉(q0.9 < q0.5),最後對三個分位數做 sort 修正。

### 7.5 `evaluate.py`
```python
import numpy as np
import polars as pl

def metrics(actual, pred):
    err = pred - actual
    ae = np.abs(err)
    return {
        "mae": ae.mean(),
        "rmse": np.sqrt((err**2).mean()),
        "mean_error": err.mean(),
        "p50": np.percentile(ae, 50),
        "p80": np.percentile(ae, 80),
        "p90": np.percentile(ae, 90),   # ← 主指標
        "p95": np.percentile(ae, 95),
        "within_60":  (ae <= 60).mean()*100,
        "within_120": (ae <= 120).mean()*100,
        "overpromise_gt60":  (err > 60).mean()*100,   # 實際比預測晚(壞)
        "underpromise_gt60": (err < -60).mean()*100,  # 實際比預測早
    }

# 主結果表:baseline(raw driver_eta) vs Linear vs Ridge vs LightGBM vs Quantile
# 切片分析:按 start_town / 小時 / is_special_day / 天氣 分組各算一次 metrics
# 校準圖:預測 ETA 分箱,畫每箱平均實際時間,理想貼 45 度線
# coverage:分位數的 90% 區間,實際落在裡面的比例是否 ~90%
```

---

## 8. 加分項與待補資料

### 8.1 台灣特殊日特徵(高 CP 值)
- 做一張 **2026 台灣假日 + 節慶表**:春節(2/14~2/22 連假,除夕 2/17)、228、清明、端午、中秋、雙十、各連假、跨年;若可能再加颱風天。
- 因為 `2/16~2/19` 已證實「低量 + 高誤差」,`is_special_day` 在這些日子會很有用,且別組想不到。

### 8.2 叫車量(供需鬆緊代理)
- 從資料自算:某 `start_town` × 某小時的歷史平均單量。量少 → 司機稀疏 → 往上修更多。

### 8.3 天氣(中央氣象署開放資料)
- 用 `start_county` + `request_time`(到小時/日)join 降雨量。雨天 ETA 系統性偏高,加這個幾乎一定有效。
- 注意:這是獨立模組,不卡主 pipeline 接口,可平行做。

---

## 9. 重要陷阱與限制(務必讀,寫報告也要誠實交代)

1. **沒有司機接單當下位置**:資料只有上車/下車點,沒有司機起點。所以無法重建「司機開來的那段路」幾何 → `driver_eta` 本身一定要當最重要特徵(已 log 後放入)。
2. **沒有「接單時間」**:只有 `request_time`(請求),沒有 accept 時間。我們校正的是 **accept→arrive**,**不含媒合等待**。報告要寫明,免得被質疑。
3. **`end_*` 對接駁 ETA 關聯小**:可先不用,別分心。
4. **早到 vs 晚到不對稱**:晚到(用戶乾等)比早到(提早到)傷害大。線性模型已出現「救了晚到、卻多了早到」的副作用——這其實划算,但要主動講成「我們有意識偏向避免晚到」。分位數模型可把這變成**可調的設計**(刻意選偏保守的分位數)。
5. **洩漏防線**:任何用到 label 的統計(TE、量統計)只能用 train 那段時間算;train 自身要 5-fold OOF。時間切分必須連續,不可隨機。
6. **派遣排序評估的邊界**:資料是「一行程一司機」,沒有候選司機池、沒司機起點,**無法做真正的反事實「該不該選這司機」模擬**。退路是排序 proxy(同時空 bucket 內,比較校正前後 ETA 排序與實際抵達順序的 Spearman/Kendall),並**誠實標明是 constructed proxy**。

---

## 10. 環境設定(Mac M1 / 16GB)

16GB 跑這個資料量無壓力。核心紀律:**用 Polars,不要用 pandas 硬讀 1.14GB。**

### 路線 A:uv(推薦,快)
```bash
uv venv && source .venv/bin/activate
uv pip install polars lightgbm scikit-learn pandas matplotlib jupyterlab h3
```

### 路線 B:miniforge(對 M1 最穩)
```bash
brew install miniforge
conda create -n eta python=3.11 && conda activate eta
conda install -c conda-forge polars lightgbm scikit-learn pandas matplotlib jupyterlab h3-py
```

- 讀檔一律 `pl.scan_parquet(...).collect()`(lazy)。
- LightGBM 在此資料量 CPU 幾分鐘跑完,**不需要 GPU、不需要雲端**。

---

## 11. 建議執行順序(給 Claude Code 的工作流)

```
1. 跑 load_clean.py,印 target 分布(log-ratio 與加法殘差),確認系統性低估(>0)。
2. 跑 features.py,先做 H3 + 時間 + 基礎 TE(train 段先用偷懶版),再升級成 5-fold OOF。
3. 跑 train.py:先重現 baseline(raw driver_eta)→ 確認 test 上 MAE~123.6/RMSE~246.6。
4. 訓練 LightGBM(log-ratio target),在 test 上比 baseline 與 Linear/Ridge。
   目標:RMSE、P90、P95 明顯優於 Linear。印 feature_importances。
5. 訓練分位數模型,做 coverage 與「寧早勿晚」分析。
6. evaluate.py:主結果表 + 校準圖 + 切片(town/時段/特殊日/天氣)。
7. 加分:特殊日表、叫車量、天氣特徵,逐一加入看增益。
```

**主線(baseline → LightGBM → 評估)沒跑通前,不要先做加分項。**

---

## 12. 報告主結果表模板(填數字用)

| Model | MAE↓ | RMSE↓ | Mean Err | Within60↑ | Within120↑ | P90↓ | OverPromise>60↓ | MAE Imp | RMSE Imp |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (driver_eta) | 123.6 | 246.6 | +77.0 | 47.8% | 72.8% | (補) | 38.3% | 0% | 0% |
| Linear Regression | 108.0 | 173.8 | +1.1 | 43.9% | 72.7% | (補) | 25.6% | 12.6% | 29.5% |
| Ridge Regression | 108.0 | 173.8 | +1.1 | 43.9% | 72.8% | (補) | 25.6% | 12.6% | 29.5% |
| **LightGBM** ⭐ | | | | | | | | | |
| Quantile (q50) | | | | | | | | | |

> 招牌賣點:**RMSE 改善 > MAE 改善 = 成功壓制長尾大誤差**(用戶體驗崩壞的元兇)。一定要在簡報粗體標出。

---

*文件結束。如有資料/欄位變動,先更新本文件第 2、4 節,再動 code。*
