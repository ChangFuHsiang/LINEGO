"""Orchestrator: run mix pipeline end-to-end."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    from load_clean import main as step_clean
    from features   import main as step_features
    from train       import main as step_train
    from evaluate    import main as step_evaluate

    print("=" * 60)
    print("  mix/ Pipeline (B: enhanced)")
    print("=" * 60)

    t0_total = time.time()
    for label, fn in [
        ("1/4  load & clean",  step_clean),
        ("2/4  build features", step_features),
        ("3/4  train models",   step_train),
        ("4/4  evaluate",       step_evaluate),
    ]:
        print(f"\n{'─'*60}\n  {label}\n{'─'*60}")
        t0 = time.time()
        fn()
        print(f"\n  Done in {time.time()-t0:.1f}s")

    print(f"\n{'='*60}")
    print(f"  mix pipeline complete in {time.time()-t0_total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
