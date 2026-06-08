"""Assemble the .txt analysis report (+ optional JSON sidecar)."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from eta_pipeline.feature_search import SearchResult
from eta_pipeline.tuning import TuningResult


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "n/a"


def _lib_versions() -> str:
    import sklearn, optuna, yaml
    return (
        f"python {platform.python_version()} / "
        f"lightgbm {lgb.__version__} / "
        f"optuna {optuna.__version__} / "
        f"numpy {np.__version__} / "
        f"pandas {pd.__version__} / "
        f"sklearn {sklearn.__version__}"
    )


def _divider(char="=", width=72) -> str:
    return char * width


def _section(title: str) -> str:
    return f"\n{'-' * 16} {title} {'-' * (54 - len(title))}\n"


def _fmt_metrics(m: dict, indent: int = 2) -> List[str]:
    sp = " " * indent
    lines = [
        f"{sp}MAE:          {m['mae']:.1f}s",
        f"{sp}RMSE:         {m['rmse']:.1f}s",
        f"{sp}Bias:         {m['bias']:+.1f}s",
        f"{sp}Within ±60s:  {m['within_60']:.1f}%",
        f"{sp}Within ±120s: {m['within_120']:.1f}%",
        f"{sp}P50 abs err:  {m['p50']:.1f}s",
        f"{sp}P90 abs err:  {m['p90']:.1f}s",
        f"{sp}P95 abs err:  {m['p95']:.1f}s",
        f"{sp}Over-promise (late>60s): {m['over_promise']:.1f}%",
        f"{sp}Under-promise (early>60s): {m['under_promise']:.1f}%",
    ]
    if "mae_imp_pct" in m:
        lines += [
            f"{sp}MAE  improvement vs raw: {m['mae_imp_pct']:+.1f}%",
            f"{sp}RMSE improvement vs raw: {m['rmse_imp_pct']:+.1f}%",
        ]
    return lines


def _table(rows: List[dict], cols: List[str], widths: List[int]) -> List[str]:
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    lines = [fmt.format(*cols), fmt.format(*["-" * w for w in widths])]
    for row in rows:
        lines.append(fmt.format(*[str(row.get(c, ""))[:w] for c, w in zip(cols, widths)]))
    return lines


def write_report(
    *,
    run_id: str,
    cfg,
    audit_info: dict,
    search_result: Optional[SearchResult],
    tuning_result: Optional[TuningResult],
    final_metrics: dict,
    raw_metrics: dict,
    segment_results: Dict[str, Any],
    importance_df: Optional[pd.DataFrame],
    device_info: str,
    artifacts_dir: Path,
    reports_dir: Path,
    cli_cmd: str = "",
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"run_{run_id}.txt"

    lines: List[str] = []
    add = lines.append

    # ── Header ──────────────────────────────────────────────────
    add(_divider())
    add(" ETA CORRECTION — RUN REPORT")
    add(_divider())
    add(f"Run id            : {run_id}")
    add(f"Timestamp / host  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  /  {socket.gethostname()}")
    add(f"Git commit        : {_git_sha()}")
    add(f"Library versions  : {_lib_versions()}")
    add(f"Device            : {device_info}")
    add(f"Random seed       : {cfg.seed}")

    # ── 1. Data & Splits ────────────────────────────────────────
    add(_section("1. DATA & SPLITS"))
    for sp, n in audit_info.get("rows", {}).items():
        add(f"  {sp:6s}: {n:>10,} rows")
    add(f"  LNY window     : {audit_info.get('lny_window', 'n/a')}")
    add(f"  LNY rows(train): {audit_info.get('lny_rows_train', 'n/a')}")
    add(f"  Clip rule      : {audit_info.get('clip_rule', 'none')}")

    if "eta_error_stats" in audit_info:
        es = audit_info["eta_error_stats"]
        add(f"\n  eta_error (train): mean={es['mean']:+.1f}s  std={es['std']:.1f}s  "
            f"p10={es['p10']:+.1f}s  p50={es['p50']:+.1f}s  p90={es['p90']:+.1f}s")

    if "driver_eta_stats" in audit_info:
        ds = audit_info["driver_eta_stats"]
        add(f"  driver_eta (train): mean={ds['mean']:.1f}s  p50={ds['p50']:.1f}s  p90={ds['p90']:.1f}s")

    null_pct = audit_info.get("null_pct_train", {})
    if null_pct:
        add(f"\n  Null % in train:")
        for col, pct in null_pct.items():
            add(f"    {col}: {pct:.2f}%")

    # ── 2. Config Snapshot ──────────────────────────────────────
    add(_section("2. CONFIG SNAPSHOT"))
    add(f"  Phases:           {cfg.phases}")
    add(f"  Core blocks:      {list(cfg.features.core_blocks)}")
    add(f"  Candidate blocks: {list(cfg.features.candidate_blocks)}")
    add(f"  Search strategy:  {cfg.search.strategy}  (max_blocks={cfg.search.max_blocks})")
    add(f"  Tuning:           n_trials={cfg.tuning.n_trials}  timeout={cfg.tuning.timeout_seconds}s")
    add(f"  Model objective:  {cfg.model.objective}")

    # ── 3. Feature Search ───────────────────────────────────────
    add(_section("3. FEATURE SEARCH"))
    if search_result is None:
        add("  (skipped)")
    else:
        add(f"  Strategy: {search_result.strategy}")
        add(f"  Best block set: {search_result.best_blocks}")
        add(f"  Best columns ({len(search_result.best_columns)}): {search_result.best_columns}")
        add(f"\n  Leaderboard (top 10):")
        lb = search_result.leaderboard[:10]
        for i, e in enumerate(lb):
            add(f"  {i+1:2d}. val_MAE={e.val_mae:.2f}s  Δcore={e.delta_vs_core:+.2f}s  "
                f"#feat={e.n_features}  {e.blocks}")

        if search_result.ablation:
            add(f"\n  Ablation (marginal contribution per block):")
            for e in search_result.ablation:
                missing = set(search_result.best_blocks) - set(e.blocks)
                blk = list(missing)[0] if missing else "?"
                add(f"    -{blk:30s}  → val_MAE={e.val_mae:.2f}s  Δ={e.delta_vs_core:+.2f}s")

    # ── 4. Hyperparameter Tuning ────────────────────────────────
    add(_section("4. HYPERPARAMETER TUNING"))
    if tuning_result is None:
        add("  (skipped)")
    else:
        add(f"  Trials run     : {tuning_result.n_trials}")
        add(f"  Best val MAE   : {tuning_result.val_mae:.2f}s")
        add(f"  Best iteration : {tuning_result.best_iteration}")
        add(f"  Best params    :")
        for k, v in tuning_result.best_params.items():
            if k not in ("objective", "metric", "verbosity", "device", "seed"):
                add(f"    {k}: {v}")
        if tuning_result.param_importances:
            add(f"\n  Param importances:")
            for k, v in sorted(tuning_result.param_importances.items(), key=lambda x: -x[1]):
                add(f"    {k:30s}: {v:.4f}")

    # ── 5. Final Model — Holdout ────────────────────────────────
    add(_section("5. FINAL MODEL — TEST HOLDOUT"))
    raw_m = raw_metrics
    fin_m = final_metrics
    hdr = f"  {'Metric':30s} {'Raw API':>12} {'Corrected':>12} {'Delta':>10}"
    add(hdr)
    add("  " + "-" * 66)
    metrics_to_show = [
        ("MAE (s)",           "mae"),
        ("RMSE (s)",          "rmse"),
        ("Bias (s)",          "bias"),
        ("Within ±60s %",     "within_60"),
        ("Within ±120s %",    "within_120"),
        ("P50 abs err (s)",   "p50"),
        ("P90 abs err (s)",   "p90"),
        ("P95 abs err (s)",   "p95"),
        ("Over-promise %",    "over_promise"),
        ("Under-promise %",   "under_promise"),
    ]
    for label, key in metrics_to_show:
        rv = raw_m.get(key, float("nan"))
        fv = fin_m.get(key, float("nan"))
        dv = fv - rv if not (np.isnan(rv) or np.isnan(fv)) else float("nan")
        add(f"  {label:30s} {rv:>12.2f} {fv:>12.2f} {dv:>+10.2f}")

    # ── 6. Segment Analysis ─────────────────────────────────────
    add(_section("6. SEGMENT ANALYSIS"))
    if not segment_results:
        add("  (no segment results)")
    else:
        for by, segs in segment_results.items():
            add(f"\n  By {by}:")
            add(f"  {'Group':25s} {'n':>7} {'Raw MAE':>9} {'Corr MAE':>10} {'Imp %':>8}")
            add("  " + "-" * 64)
            for val, m in sorted(segs.items(), key=lambda x: str(x[0])):
                raw_mae = m.get("mae", 0) - m.get("mae_vs_raw", 0)
                imp = m.get("mae_imp_pct", 0)
                flag = " ⚠" if m.get("mae_vs_raw", 0) > 0 else ""
                add(f"  {str(val):25s} {m['n']:>7,} {raw_mae:>9.1f} {m['mae']:>10.1f} "
                    f"{imp:>7.1f}%{flag}")

    # ── 7. Feature Importance ───────────────────────────────────
    add(_section("7. FEATURE IMPORTANCE"))
    if importance_df is None:
        add("  (not available)")
    else:
        add(f"  {'Feature':35s} {'Gain':>10}")
        add("  " + "-" * 47)
        for _, row in importance_df.head(20).iterrows():
            add(f"  {row['feature']:35s} {row['importance']:>10.0f}")

    # ── 8. Artifacts & Reproducibility ──────────────────────────
    add(_section("8. ARTIFACTS & REPRODUCIBILITY"))
    add(f"  Artifacts dir : {artifacts_dir}")
    add(f"  Reports dir   : {reports_dir}")
    add(f"  Model file    : {artifacts_dir / f'model_{run_id}.lgb'}")
    add(f"  Study db      : {artifacts_dir / f'study_{run_id}.db'}")
    add(f"  CLI to reproduce: {cli_cmd}")
    add(_divider())

    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    print(f"\nReport written: {report_path}")

    if cfg.report.emit_json_sidecar:
        sidecar = {
            "run_id": run_id,
            "raw_metrics": raw_metrics,
            "final_metrics": final_metrics,
            "best_blocks": search_result.best_blocks if search_result else None,
            "best_params": tuning_result.best_params if tuning_result else None,
        }
        sidecar_path = reports_dir / f"run_{run_id}.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"JSON sidecar : {sidecar_path}")

    return report_path
