"""
Run full PKU recommender pipeline: Stage 0 → 2c → 1 → 5-fold CV.

Recommended order:
  1. Stage 0    – Data preparation (ratings, food DB, etc.)
  2. Stage 2c   – Meal clustering (full dataset; not split-dependent)
  3. Stage 1    – Train/test split (80/20)
  4. 5-fold CV  – Robustness experiment; then restores 80/20 and runs Stages 2–7

So "5-fold" runs after the split is created (Stage 1). It does its CV, then
restores the 80/20 split and runs Stages 2, 3, 4, 5, 6, 7 so all outputs
(recommendations, evaluations, stats, figures) are updated for the main split.

Usage: python run_full_pipeline.py
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

    print("\n" + "="*60)
    print("FULL PIPELINE: Stage 0 → 2c → 1 → 5-fold (then 2–7 inside 5-fold)")
    print("="*60)

    steps = [
        ("stage0_data_preparation.py", "Stage 0 – Data preparation"),
        ("stage2c_meal_clustering.py", "Stage 2c – Meal clustering"),
        ("stage1_train_test_split.py", "Stage 1 – Train/test split (80/20)"),
        ("run_5fold_cross_validation.py", "5-fold CV + restore 80/20 + Stages 2–7"),
    ]

    for script, desc in steps:
        if not run_script(script, desc):
            print(f"\nStopping: fix the failure above and re-run.")
            sys.exit(1)

    print("\n" + "="*60)
    print("Pipeline complete. Check cross_validation_5fold_results.csv and")
    print("all stage outputs (preference/safety CSVs, visualizations).")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
