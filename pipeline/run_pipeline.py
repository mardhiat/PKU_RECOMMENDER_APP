import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pku.paths import paths

"""
Run the evaluation pipeline from the repository root or from pipeline/.

Order:
  1. prepare_data.py     ratings fixture, meal table, user limits
  2. cluster_meals.py    clusters on the full meal catalog (not split-dependent)
  3. split_ratings.py    80/20 user-wise split
  4. cross_validate.py   5-fold check, then restores 80/20 and runs recommend through visualize

Clustering runs before the split because clusters describe foods, not users.
Parameter search is separate: python pipeline/tune_parameters.py
"""

import os
import sys
import subprocess


def check_dependencies():
    """Ensure required packages are installed; exit with clear message if not."""
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        print("Missing dependencies. Install with ONE of:")
        print("  pip install -r requirements-pipeline.txt   (pipeline only, no Rust)")
        print("  pip install -r requirements.txt            (full app; needs Python 3.11/3.12 or Rust)")
        print("")
        print("If pip install fails (e.g. pydantic/Rust on Python 3.14), use requirements-pipeline.txt")
        print("or switch to Python 3.11 or 3.12 and use requirements.txt.")
        sys.exit(1)


def run_script(script_name, description):
    """Run a Python script; return True on success."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, script_name)
    if not os.path.exists(path):
        print(f"  Skip: {description} (not found: {script_name})")
        return False
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    r = subprocess.run(
        [sys.executable, path],
        cwd=cwd,
    )
    if r.returncode != 0:
        print(f"  FAILED: {description} (exit code {r.returncode})")
        return False
    print(f"  OK: {description}")
    return True


def main():
    check_dependencies()

    print("Pipeline: prepare -> cluster -> split -> cross-validate")

    steps = [
        ("prepare_data.py", "Prepare data"),
        ("cluster_meals.py", "Cluster meals"),
        ("split_ratings.py", "Train/test split (80/20)"),
        ("cross_validate.py", "5-fold check, then restore 80/20 and evaluate"),
    ]

    for script, desc in steps:
        if not run_script(script, desc):
            print(f"\nStopping: fix the failure above and re-run.")
            sys.exit(1)

    print("\n" + "="*60)
    print("Pipeline complete. Aggregates are in docs/results/.")
    print("Figures are in docs/figures/.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
