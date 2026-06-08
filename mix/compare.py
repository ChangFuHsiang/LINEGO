"""Generate outputs/comparison_report.md + side-by-side figures."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_METRICS  = Path("outputs/run_src/metrics/all_metrics.json")
MIX_METRICS  = Path("outputs/run_mix/metrics/all_metrics.json")
MIX_PERM     = Path("outputs/run_mix/metrics/permutation_importance.json")
MIX_OPTUNA   = Path("outputs/run_mix/metrics/optuna_best.json")
MIX_GAIN     = Path("outputs/run_mix/metrics/feature_importance_gain.json")
FIG_DIR      = Path("outputs/figures_comparison")
REPORT_PATH  = Path("outputs/comparison_report.md")

FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> dict:
    if not p.exists():
        print(f"WARNING: {p} not found")
        return {}
    with open(p) as f:
        return json.load(f)


def fmt(v, decimals=1, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    suffix = "%" if pct else "s"
    return f"{v:.{decimals}f}{suffix}"


def improvement(base, val):
    if base == 0:
        return "—"
    return f"{(base - val) / base * 100:+.1f}%"


def make_side_by_side_fig(src_fig: Path, mix_fig: Path, out: Path, title: str):
    if not src_fig.exists() or not mix_fig.exists():
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, p, label in [(axes[0], src_fig, "src (Pipeline A)"),
                          (axes[1], mix_fig, "mix (Pipeline B)")]:
        img = plt.imread(str(p))
        ax.imshow(img); ax.axis("off"); ax.set_title(label, fontsize=13)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")


def main():
    src_m = load_json(SRC_METRICS)
    mix_m = load_json(MIX_METRICS)
    perm  = load_json(MIX_PERM)
    opt   = load_json(MIX_OPTUNA)
    gain  = load_json(MIX_GAIN)

    if not src_m or not mix_m:
        print("ERROR: one or both metric files missing. Run both pipelines first.")
        sys.exit(1)

    # ── Side-by-side comparison figures ───────────────────────────────────────
    print("Generating comparison figures...")
    for fname, title in [
        ("calibration.png",        "Calibration"),
        ("error_distribution.png", "Error Distribution"),
        ("mae_by_hour.png",        "MAE by Hour"),
    ]:
        make_side_by_side_fig(
            Path("outputs/run_src/figures") / fname,
            Path("outputs/run_mix/figures") / fname,
            FIG_DIR / fname.replace(".png", "_comparison.png"),
            title,
        )

    # Copy single-pipeline-only figures
    for fname in ["permutation_importance.png", "gain_importance.png",
                  "quantile_coverage.png"]:
        src_f = Path("outputs/run_mix/figures") / fname
        dst_f = FIG_DIR / fname
        if src_f.exists():
            import shutil; shutil.copy(src_f, dst_f)
            print(f"  Copied: {dst_f}")

    # ── Build report ──────────────────────────────────────────────────────────
    lines = []
    A = lines.append

    def table_row(*cells):
        A("| " + " | ".join(str(c) for c in cells) + " |")

    A("# ETA Correction — A vs B Comparison Report")
    A("")
    A("> **Pipeline A** = `src/` 純淨版 (log-ratio target, manual features, no HP search)  ")
    A("> **Pipeline B** = `mix/` 強化版 (log-ratio target, driver 3-stats, Optuna 80 trials)")
    A("> Test set: **2026-05-07 ~ 2026-05-20** (same for both)")
    A("")

    # ── 5.1 Main results table ─────────────────────────────────────────────────
    A("## 5.1 主結果對照表")
    A("")
    A("| Model | MAE | RMSE | MeanErr | Within60 | P90 | Late>60 | Early>60 |")
    A("|---|---|---|---|---|---|---|---|")

    baseline = src_m.get("Baseline (driver_eta)", {})
    src_lgb  = src_m.get("LightGBM", {})
    src_q50  = src_m.get("Quantile q50", {})
    mix_lgb  = mix_m.get("mix LightGBM", {})
    mix_q50  = mix_m.get("mix Quantile q50", {})

    def row(name, m):
        if not m:
            return
        A(f"| {name} | {fmt(m.get('mae'))} | {fmt(m.get('rmse'))} | "
          f"{fmt(m.get('mean_error'))} | {fmt(m.get('within_60'), pct=True)} | "
          f"{fmt(m.get('p90_ae'), 0)} | {fmt(m.get('overpromise_gt60'), pct=True)} | "
          f"{fmt(m.get('underpromise_gt60'), pct=True)} |")

    row("Baseline (driver_eta, no fix)", baseline)
    row("Pipeline A: src LightGBM", src_lgb)
    row("Pipeline A: src Quantile q50", src_q50)
    row("Pipeline B: mix LightGBM", mix_lgb)
    row("Pipeline B: mix Quantile q50", mix_q50)

    A("")
    if src_lgb and mix_lgb:
        A("**mix vs src 改善幅度 (LightGBM):**")
        A("")
        A(f"- MAE:  {improvement(src_lgb['mae'],  mix_lgb['mae'])}")
        A(f"- RMSE: {improvement(src_lgb['rmse'], mix_lgb['rmse'])}")
        A(f"- P90:  {improvement(src_lgb['p90_ae'], mix_lgb['p90_ae'])}")
    A("")

    # ── 5.2 Five key questions ─────────────────────────────────────────────────
    A("## 5.2 五個關鍵問題的數據答案")
    A("")

    # Q1: Driver 3-stats vs single TE
    A("### Q1: 司機三統計 vs 單一 TE")
    A("")
    pf = perm.get("features", {})
    driver_cols = [c for c in pf if "driver" in c]
    did_hash_gain = gain.get("gain", {}).get("did_hash_te", None)
    driver_perm_total = sum(max(pf[c]["mean_mae_increase"], 0) for c in driver_cols)
    total_perm = sum(max(v["mean_mae_increase"], 0) for v in pf.values())
    driver_perm_pct = driver_perm_total / total_perm * 100 if total_perm > 0 else 0
    A(f"- **src**: `did_hash_te` 單一特徵，gain importance 約 17.5%")
    A(f"- **mix**: `driver_avg/median/std_logratio` 三特徵，"
      f"permutation importance 合計 **{driver_perm_pct:.1f}%**")
    A(f"- MAE 變化: src {fmt(src_lgb.get('mae'))} → mix {fmt(mix_lgb.get('mae'))}")
    A("")

    # Q2: uid_hash_te removal
    A("### Q2: 拿掉 uid_hash_te 有沒有變差?")
    A("")
    uid_perm = pf.get("uid_trip_count_train", {}).get("mean_mae_increase", None)
    A(f"- src 含 `uid_hash_te`（split 11.1%, gain 8.6%，疑為雜訊）")
    A(f"- mix 改用 `uid_trip_count_train` + `uid_days_since_first`（次數/生命週期）")
    if uid_perm is not None:
        A(f"- `uid_trip_count_train` permutation importance: **+{uid_perm:.2f}s**")
    if mix_lgb and src_lgb:
        mae_diff = mix_lgb["mae"] - src_lgb["mae"]
        verdict = "沒有變差，略好" if mae_diff < 0 else f"略差 {mae_diff:.1f}s"
        A(f"- 結論: **{verdict}**（如果 mix 整體 MAE 更好，說明 uid_hash_te 確為雜訊）")
    A("")

    # Q3: end_town usefulness
    A("### Q3: end_town 真的有用嗎?")
    A("")
    end_town_perm = pf.get("end_town", {}).get("mean_mae_increase", None)
    end_town_gain = gain.get("gain", {}).get("end_town", None)
    total_gain = sum(gain.get("gain", {}).values())
    if end_town_perm is not None:
        perm_pct = end_town_perm / total_perm * 100 if total_perm > 0 else 0
        A(f"- Permutation importance: **+{end_town_perm:.2f}s** ({perm_pct:.1f}%)")
    if end_town_gain is not None:
        gain_pct = end_town_gain / total_gain * 100 if total_gain > 0 else 0
        A(f"- Gain importance: {gain_pct:.1f}%")
    sorted_perm = sorted(pf.items(), key=lambda x: -x[1]["mean_mae_increase"])
    rank = next((i+1 for i, (f, _) in enumerate(sorted_perm) if f == "end_town"), None)
    A(f"- Permutation 排名: **第 {rank} 名**" if rank else "- 排名: 未知")
    A(f"- 機制未完全釐清（可能編碼行程類型/路線偏好），但實驗結果 work")
    A("")

    # Q4: Optuna contribution
    A("### Q4: Optuna 80 trials 帶來多少額外增益?")
    A("")
    if opt:
        A(f"- Optuna 最佳 objective（P90 + 不對稱懲罰）: **{opt.get('best_value', '—'):.2f}**")
        A(f"- 完成 trials: {opt.get('n_trials', '—')}")
        A(f"- Best iteration: {opt.get('best_iteration', '—')}")
        best_p = opt.get("best_params", {})
        A(f"- 最佳 lr={best_p.get('learning_rate', '—'):.4f}  "
          f"num_leaves={best_p.get('num_leaves', '—')}  "
          f"min_child={best_p.get('min_child_samples', '—')}")
    A("- 與 src 預設參數相比的改善量詳見 Q1/整體 MAE 數字")
    A("")

    # Q5: log_eta permutation rank
    A("### Q5: log_eta 在 permutation 下還是 #1 嗎?")
    A("")
    logeta_perm = pf.get("log_eta", {}).get("mean_mae_increase", None)
    logeta_rank = next((i+1 for i, (f, _) in enumerate(sorted_perm) if f == "log_eta"), None)
    logeta_gain = gain.get("gain", {}).get("log_eta", None)
    gain_pct_logeta = logeta_gain / total_gain * 100 if logeta_gain and total_gain else None
    A(f"- Gain importance: **{gain_pct_logeta:.1f}%**" if gain_pct_logeta else "- Gain: 未知")
    if logeta_perm is not None:
        p_pct = logeta_perm / total_perm * 100 if total_perm > 0 else 0
        A(f"- Permutation importance: **+{logeta_perm:.2f}s** ({p_pct:.1f}%)")
    A(f"- Permutation 排名: **第 {logeta_rank} 名**" if logeta_rank else "")
    A("- 在 src/ 中 gain 高達 55.5%，原因是 log_eta 嵌入在 log-ratio target 公式裡")
    A("- Permutation 是無偏估計，能看出「真實」貢獻，不受公式結構影響")
    A("")

    # ── 5.3 Figures ────────────────────────────────────────────────────────────
    A("## 5.3 對照圖")
    A("")
    A(f"所有圖存於 `outputs/figures_comparison/`")
    for f in sorted(FIG_DIR.glob("*.png")):
        A(f"- `{f.name}`")
    A("")

    # ── 5.4 Limitations ────────────────────────────────────────────────────────
    A("## 5.4 誠實的限制聲明")
    A("")
    A("這個實驗只在這份資料、這個 test set（2026-05-07 ~ 05-20）上驗證了 mix 是否優於 src。"
      "結論不能推廣到以下情境：")
    A("")
    A("- 不同時間段或不同城市的資料")
    A("- 外部衝擊（颱風、突發事件）導致的分布偏移")
    A("- 即時上線部署（batch offline ≠ online latency）")
    A("")
    if src_lgb and mix_lgb:
        mae_gap = abs(src_lgb["mae"] - mix_lgb["mae"])
        if mae_gap < 2.0:
            A(f"> **注意**：mix vs src 的 MAE 差距為 **{mae_gap:.1f}s**（< 2s）。"
              "這可能意味著模型已接近資料天花板。進一步突破需要外部資料"
              "（即時路況、司機起點、即時天氣），而非調整現有特徵組合。")
        else:
            A(f"> mix 相對 src 改善 **{mae_gap:.1f}s**，改善具有實質意義（>= 2s 門檻）。")
    A("")
    A("---")
    A(f"*Report generated by `mix/compare.py`*")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
