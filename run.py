"""
Entry point for the full ETA correction pipeline.

Creates a timestamped run directory under outputs/ and runs all four
stages in sequence. All intermediate files and results are written there.

Usage:
    python run.py                          # log-ratio target (default)
    python run.py --target additive        # additive residual target
    python run.py --run-dir outputs/debug  # custom run directory
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import new_run_dir
from load_clean import main as step_clean
from features   import main as step_features
from train       import main as step_train
from evaluate    import main as step_evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target",  default="target_logratio",
                   choices=["target_logratio", "target_additive"],
                   help="Which residual target to use (default: target_logratio)")
    p.add_argument("--run-dir", default=None,
                   help="Override the output directory (default: outputs/<timestamp>)")
    return p.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  ETA Correction Pipeline")
    print(f"  Run dir : {run_dir}")
    print(f"  Target  : {args.target}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    steps = [
        ("1/4  load & clean",  lambda: step_clean(run_dir=run_dir)),
        ("2/4  build features", lambda: step_features(run_dir=run_dir, target_col=args.target)),
        ("3/4  train models",   lambda: step_train(run_dir=run_dir, target_col=args.target)),
        ("4/4  evaluate",       lambda: step_evaluate(run_dir=run_dir, target_col=args.target)),
    ]

    total_start = time.time()
    for label, fn in steps:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        t0 = time.time()
        fn()
        print(f"\n  Done in {time.time() - t0:.1f}s")

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
