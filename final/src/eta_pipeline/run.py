"""CLI orchestrator: load config → data → features → search → tune → final → report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="eta-run",
        description="ETA correction pipeline",
    )
    p.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config (default: config/default.yaml)",
    )
    p.add_argument(
        "--phase", default="all",
        choices=["all", "feature", "tune", "final"],
        help="Run only this phase (default: all). Use --run-id to resume.",
    )
    p.add_argument(
        "--run-id", default=None,
        help="Reuse an existing run_id to resume from a previous phase.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config.",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Shortcut: use config/quick.yaml instead of default.",
    )
    p.add_argument(
        "--work-dir", default=".",
        help="Working directory for relative paths (default: current dir).",
    )
    return p.parse_args(argv)


def _artifacts(run_id: str, work_dir: Path) -> Path:
    p = work_dir / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _phase_file(artifacts: Path, phase: str, run_id: str) -> Path:
    return artifacts / f"{phase}_{run_id}.json"


def main(argv=None):
    args = parse_args(argv)
    work_dir = Path(args.work_dir).resolve()

    # Resolve config: check work_dir, then final/, then package root
    pkg_root = Path(__file__).parent.parent.parent  # final/
    _cfg_name = "config/quick.yaml" if args.quick else args.config
    for base in [work_dir, pkg_root]:
        candidate = base / _cfg_name
        if candidate.exists():
            config_path = candidate
            break
    else:
        config_path = Path(_cfg_name)

    # Lazy imports so the module can be imported without heavy deps
    from eta_pipeline.config import load_config
    from eta_pipeline.model import detect_device

    cfg = load_config(config_path)

    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_id is not None:
        cfg.run_id = args.run_id
    else:
        # Re-generate run_id now that seed may have changed
        from eta_pipeline.config import RunConfig
        import hashlib, json as _json, dataclasses
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        h  = hashlib.md5(
            _json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        cfg.run_id = f"{ts}-{h}"

    run_id   = cfg.run_id
    artifacts = _artifacts(run_id, work_dir)
    reports_dir = work_dir / cfg.report.dir
    phases_to_run = cfg.phases if args.phase == "all" else [args.phase]

    device, device_info = detect_device(cfg.device)

    cli_cmd = f"python -m eta_pipeline.run --config {config_path} --run-id {run_id}"

    print("=" * 72)
    print(f"  ETA Correction Pipeline")
    print(f"  Run id  : {run_id}")
    print(f"  Config  : {config_path}")
    print(f"  Device  : {device_info}")
    print(f"  Phases  : {phases_to_run}")
    print("=" * 72)

    # ── Data ────────────────────────────────────────────────────────────────
    from eta_pipeline.data import load_raw, clean, add_target, make_splits, audit as run_audit
    import numpy as np

    print("\n[Data] Loading and cleaning…")
    raw_pl = load_raw(work_dir / cfg.data.path)
    df_pl  = clean(raw_pl, cfg)
    df_pl  = add_target(df_pl, cfg)

    print("[Data] Splitting…")
    splits = make_splits(df_pl, cfg)
    audit_info = run_audit(splits, cfg)

    # ── Features (superset, built once) ─────────────────────────────────────
    from eta_pipeline.features import build_features, REGISTRY
    from eta_pipeline.feature_search import _block_cols

    all_block_names = cfg.all_blocks
    print(f"\n[Features] Building superset for blocks: {all_block_names}")
    t0 = time.time()
    bundles = build_features(all_block_names, splits, cfg)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"X shape: {bundles['train'].X.shape}")

    # ── Phase A: Feature search ──────────────────────────────────────────────
    search_result = None
    selected_file = _phase_file(artifacts, "selected", run_id)

    if "feature" in phases_to_run:
        from eta_pipeline.feature_search import run_search
        print("\n[Phase A] Feature search…")
        t0 = time.time()
        search_result = run_search(
            bundles,
            list(cfg.features.core_blocks),
            list(cfg.features.candidate_blocks),
            cfg,
        )
        print(f"  Done in {time.time()-t0:.1f}s")
        payload = {
            "blocks":      search_result.best_blocks,
            "columns":     search_result.best_columns,
            "categorical": search_result.cat_cols,
        }
        selected_file.write_text(json.dumps(payload, indent=2))
        print(f"  Saved selected features → {selected_file}")
    elif selected_file.exists():
        payload = json.loads(selected_file.read_text())
        print(f"[Phase A] Skipped — loaded from {selected_file}")
        from eta_pipeline.feature_search import SearchResult
        search_result = SearchResult(
            strategy="loaded",
            best_blocks=payload["blocks"],
            best_columns=payload["columns"],
            cat_cols=payload["categorical"],
            leaderboard=[],
        )
    else:
        # Fallback: use all blocks
        from eta_pipeline.feature_search import SearchResult, _block_cols, _block_cat_cols
        all_cols = _block_cols(all_block_names)
        search_result = SearchResult(
            strategy="fallback_all",
            best_blocks=all_block_names,
            best_columns=all_cols,
            cat_cols=_block_cat_cols(all_block_names, bundles["train"].cat_cols),
            leaderboard=[],
        )

    feature_cols = search_result.best_columns
    cat_cols     = search_result.cat_cols

    # ── Phase B: Hyperparameter tuning ──────────────────────────────────────
    tuning_result = None
    params_file   = _phase_file(artifacts, "best_params", run_id)

    if "tune" in phases_to_run:
        from eta_pipeline.tuning import run_tuning
        print("\n[Phase B] Hyperparameter tuning…")
        t0 = time.time()
        tuning_result = run_tuning(
            bundles, feature_cols, cat_cols, cfg,
            artifacts_dir=artifacts, run_id=run_id, device=device,
        )
        print(f"  Done in {time.time()-t0:.1f}s")
        params_payload = {
            "params":         tuning_result.best_params,
            "best_iteration": tuning_result.best_iteration,
        }
        params_file.write_text(json.dumps(params_payload, indent=2))
    elif params_file.exists():
        params_payload = json.loads(params_file.read_text())
        print(f"[Phase B] Skipped — loaded from {params_file}")
        from eta_pipeline.tuning import TuningResult
        tuning_result = TuningResult(
            best_params=params_payload["params"],
            best_iteration=params_payload["best_iteration"],
            val_mae=float("nan"),
            n_trials=0,
        )
    else:
        # Fallback: use default params from config
        from eta_pipeline.tuning import TuningResult
        tuning_result = TuningResult(
            best_params={
                "objective": cfg.model.objective,
                "metric":    cfg.model.metric,
                "verbosity": -1,
                "device":    device,
                "seed":      cfg.seed,
                "num_leaves":       63,
                "learning_rate":    0.05,
                "min_data_in_leaf": 100,
            },
            best_iteration=400,
            val_mae=float("nan"),
            n_trials=0,
        )

    # ── Phase C: Final retrain + evaluation ─────────────────────────────────
    if "final" in phases_to_run:
        from eta_pipeline import model as mdl
        from eta_pipeline.metrics import evaluate, segment_analysis
        from eta_pipeline.report import write_report

        print("\n[Phase C] Final retrain on train+val…")
        tr = bundles["train"]
        va = bundles["val"]
        te = bundles["test"]

        # Combine train + val
        import pandas as pd
        X_tv = pd.concat([tr.X, va.X], axis=0)
        y_tv = np.concatenate([tr.y, va.y])

        num_rounds = int(tuning_result.best_iteration * cfg.tuning.round_inflation)
        final_params = dict(tuning_result.best_params)
        final_params.update({"device": device})

        avail = [c for c in feature_cols if c in X_tv.columns]
        eff_cat = [c for c in cat_cols if c in avail]

        import lightgbm as lgb
        dtrain = lgb.Dataset(X_tv[avail], y_tv,
                             categorical_feature=eff_cat or "auto", free_raw_data=False)
        final_model = lgb.train(
            final_params, dtrain,
            num_boost_round=num_rounds,
            callbacks=[lgb.log_evaluation(50)],
        )

        test_df   = splits["test"]
        actual_te = test_df["time_accept_to_arrive"].values
        raw_eta   = test_df["driver_eta"].values

        pred_residual = final_model.predict(te.X[avail])
        corrected_eta = mdl.restore_eta(raw_eta, pred_residual)

        raw_metrics   = evaluate(actual_te, raw_eta.astype(float))
        final_metrics = evaluate(actual_te, corrected_eta, raw=raw_eta.astype(float))

        print(f"\n  Baseline  MAE={raw_metrics['mae']:.1f}s  RMSE={raw_metrics['rmse']:.1f}s")
        print(f"  Corrected MAE={final_metrics['mae']:.1f}s  RMSE={final_metrics['rmse']:.1f}s  "
              f"(MAE imp {final_metrics.get('mae_imp_pct', 0):.1f}%)")

        # Segment analysis
        seg_df = test_df.copy()
        seg_df["_corrected"] = corrected_eta
        seg_df["_raw"]       = raw_eta.astype(float)
        if "haversine_m" in te.X.columns:
            seg_df["haversine_m"] = te.X["haversine_m"].values

        segs = segment_analysis(seg_df, "time_accept_to_arrive", "_corrected", "_raw", cfg)

        # Feature importance
        imp_df = mdl.feature_importance(final_model, "gain")

        # Save model
        model_path = artifacts / f"model_{run_id}.lgb"
        mdl.save_model(final_model, model_path)

        # Plots
        if cfg.report.emit_plots:
            from eta_pipeline import plotting
            plotting.save_all(actual_te, corrected_eta, raw_eta.astype(float),
                              imp_df, reports_dir / f"plots_{run_id}")

        # Write report
        write_report(
            run_id=run_id,
            cfg=cfg,
            audit_info=audit_info,
            search_result=search_result,
            tuning_result=tuning_result,
            final_metrics=final_metrics,
            raw_metrics=raw_metrics,
            segment_results=segs,
            importance_df=imp_df,
            device_info=device_info,
            artifacts_dir=artifacts,
            reports_dir=reports_dir,
            cli_cmd=cli_cmd,
        )

    elapsed = time.time() - t0 if "t0" in dir() else 0
    print(f"\n{'=' * 72}")
    print(f"  Pipeline complete.  Run id: {run_id}")
    print(f"  Artifacts: {artifacts}")
    print(f"  Reports  : {reports_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
