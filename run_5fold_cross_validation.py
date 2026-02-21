"""
5-Fold Cross-Validation for PKU Recommender Evaluation (Robustness Experiment)

This script adds a robustness experiment to the evaluation pipeline:
- Instead of a single 80/20 split, we use 5-fold cross-validation.
- Every sample is used for testing exactly once (each fold uses 20% as test).
- Results are reported as mean ± std across 5 folds for increased credibility.

Run after stage0 and stage2c (data + meal clusters). Requires stage2 and stage4.

After the 5-fold run completes, the script restores the original 80/20 split
and re-runs all main pipeline stages (1 through 7) so that:
- Train/test CSVs and all recommendation/evaluation outputs are updated to the
  primary 80/20 split (not the last CV fold).
- You get both: cross_validation_5fold_results.csv (robustness) and up-to-date
  main pipeline outputs (preference summaries, stage5/6/7 results, visualizations).
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import subprocess
from sklearn.model_selection import KFold

# Configuration (match stage1)
MIN_RATINGS_PER_USER = 10
MIN_LIKED_FOODS = 3
N_FOLDS = 5
RANDOM_STATE = 42


def load_and_filter_data():
    """Load stage0 data and apply same filtering as stage1. Returns filtered_ratings, eligible_users."""
    ratings_df = pd.read_csv('data_user_food_ratings.csv')
    food_db = pd.read_csv('data_food_database.csv')

    food_db['food_name_lower'] = food_db['food_name'].str.lower().str.strip()
    ratings_df['food_lower'] = ratings_df['food'].str.lower().str.strip()

    foods_with_nutrition = food_db[
        (food_db['phe_mg_per_100g'] > 0) |
        (food_db['protein_g_per_100g'] > 0) |
        (food_db['data_quality'].isin(['complete', 'partial']))
    ]
    valid_foods_lower = set(foods_with_nutrition['food_name_lower'].values)
    filtered_ratings = ratings_df[ratings_df['food_lower'].isin(valid_foods_lower)].copy()
    filtered_ratings = filtered_ratings.drop(columns=['food_lower'])

    user_stats = filtered_ratings.groupby('user_name').agg({
        'rating': ['count', lambda x: (x >= 4).sum()]
    }).reset_index()
    user_stats.columns = ['user_name', 'total_ratings', 'liked_foods']
    eligible_users = user_stats[
        (user_stats['total_ratings'] >= MIN_RATINGS_PER_USER) &
        (user_stats['liked_foods'] >= MIN_LIKED_FOODS)
    ]['user_name'].values

    filtered_ratings = filtered_ratings[filtered_ratings['user_name'].isin(eligible_users)].copy()
    filtered_ratings = filtered_ratings.drop_duplicates(subset=['user_name', 'food'], keep='first')

    return filtered_ratings, eligible_users


def create_5fold_splits_simple(filtered_ratings, eligible_users):
    """
    Create 5 folds: for each user, assign each rating to fold 0..4 (stratified by liked/not-liked).
    Every sample is in test exactly once. Gives a more stable estimate than a single 80/20 split.
    """
    np.random.seed(RANDOM_STATE)
    folds_train = [[] for _ in range(N_FOLDS)]
    folds_test = [[] for _ in range(N_FOLDS)]

    for user_name in eligible_users:
        user_ratings = filtered_ratings[filtered_ratings['user_name'] == user_name].reset_index(drop=True)
        n = len(user_ratings)
        if n == 0:
            continue

        liked_idx = np.where(user_ratings['rating'].values >= 4)[0]
        not_liked_idx = np.where(user_ratings['rating'].values < 3)[0]
        fold_assign = np.zeros(n, dtype=int)

        # Stratified: assign liked and not-liked to folds 0..4 separately (shuffled)
        if len(liked_idx) > 0:
            shuffled = np.array(liked_idx)
            np.random.shuffle(shuffled)
            for i, pos in enumerate(shuffled):
                fold_assign[pos] = i % N_FOLDS
        if len(not_liked_idx) > 0:
            shuffled = np.array(not_liked_idx)
            np.random.shuffle(shuffled)
            for i, pos in enumerate(shuffled):
                fold_assign[pos] = i % N_FOLDS
        # (rating == 3 if any: remain in fold 0)

        for i in range(n):
            row = user_ratings.iloc[i].copy()
            f = int(fold_assign[i])
            for k in range(N_FOLDS):
                if k == f:
                    folds_test[k].append(row)
                else:
                    folds_train[k].append(row)

    result = []
    for f in range(N_FOLDS):
        train_df = pd.DataFrame(folds_train[f]).reset_index(drop=True) if folds_train[f] else pd.DataFrame(columns=filtered_ratings.columns)
        test_df = pd.DataFrame(folds_test[f]).reset_index(drop=True) if folds_test[f] else pd.DataFrame(columns=filtered_ratings.columns)
        result.append((train_df, test_df))
    return result


def _subprocess_env():
    """Use UTF-8 for subprocess I/O on Windows to avoid UnicodeEncodeError (cp1252)."""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def run_stage2():
    """Run stage2_generate_recommendations.py (reads train/test from CSV)."""
    r = subprocess.run(
        [sys.executable, 'stage2_generate_recommendations.py'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=_subprocess_env(),
    )
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError("Stage 2 failed")
    return True


def run_stage4():
    """Run stage4_evaluate_preference.py (reads recommendations from pkl)."""
    r = subprocess.run(
        [sys.executable, 'stage4_evaluate_preference.py'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=_subprocess_env(),
    )
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError("Stage 4 failed")
    return True


def run_stage(stage_script, stage_name):
    """Run a stage script; return True on success, False on failure."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(
        [sys.executable, stage_script],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd,
        env=_subprocess_env(),
    )
    if r.returncode != 0:
        print(f"  Warning: {stage_name} failed (exit {r.returncode}). Check dependencies.")
        if r.stderr:
            print(r.stderr[:500])
        return False
    return True


def extract_metrics_from_pkl():
    """Load preference_evaluation_results_TFIDF.pkl and return dict algo -> {precision_avg, recall_avg, f1_avg, hit_rate_avg}."""
    with open('preference_evaluation_results_TFIDF.pkl', 'rb') as f:
        results = pickle.load(f)
    return {algo: {k: v for k, v in res.items() if k in ('precision_avg', 'recall_avg', 'f1_avg', 'hit_rate_avg')}
            for algo, res in results.items()}


def main():
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION (Robustness Experiment)")
    print("=" * 70)
    print("\nEvery sample is used for testing exactly once. Results: mean ± std across 5 folds.\n")

    # Check dependencies
    for f in ['data_user_food_ratings.csv', 'data_food_database.csv', 'data_meal_ingredients.csv', 'data_meal_clusters.csv']:
        if not os.path.exists(f):
            print(f"ERROR: Missing {f}. Run stage0 (and stage2c for clusters) first.")
            sys.exit(1)

    # Load and filter (same as stage1)
    print("Loading and filtering data (same criteria as Stage 1)...")
    filtered_ratings, eligible_users = load_and_filter_data()
    print(f"  Eligible users: {len(eligible_users)}, Ratings: {len(filtered_ratings)}")

    # Create 5 folds
    print(f"\nCreating {N_FOLDS}-fold splits (stratified per user)...")
    folds = create_5fold_splits_simple(filtered_ratings, eligible_users)
    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i+1}: train={len(tr)}, test={len(te)}")

    # Collect metrics per fold
    algo_names = None
    fold_metrics = []  # list of dict algo -> {precision_avg, recall_avg, f1_avg, hit_rate_avg}

    for fold_i, (train_df, test_df) in enumerate(folds):
        print(f"\n--- Fold {fold_i + 1}/{N_FOLDS} ---")
        train_df.to_csv('data_train_ratings.csv', index=False)
        test_df.to_csv('data_test_ratings.csv', index=False)
        test_users = test_df['user_name'].unique()
        pd.DataFrame({'user_name': test_users}).to_csv('data_test_users.csv', index=False)

        run_stage2()
        run_stage4()
        metrics = extract_metrics_from_pkl()
        if algo_names is None:
            algo_names = list(metrics.keys())
        fold_metrics.append(metrics)

    # Aggregate: mean and std across folds
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION RESULTS (Mean ± Std)")
    print("=" * 70)

    summary_rows = []
    for algo in algo_names:
        prec = [fold_metrics[f][algo]['precision_avg'] for f in range(N_FOLDS)]
        rec = [fold_metrics[f][algo]['recall_avg'] for f in range(N_FOLDS)]
        f1 = [fold_metrics[f][algo]['f1_avg'] for f in range(N_FOLDS)]
        hr = [fold_metrics[f][algo]['hit_rate_avg'] for f in range(N_FOLDS)]
        summary_rows.append({
            'Algorithm': algo,
            'Precision@10 (%)': f"{np.mean(prec):.1f} ± {np.std(prec):.1f}",
            'Recall@10 (%)': f"{np.mean(rec):.1f} ± {np.std(rec):.1f}",
            'F1@10 (%)': f"{np.mean(f1):.1f} ± {np.std(f1):.1f}",
            'Hit Rate (%)': f"{np.mean(hr):.1f} ± {np.std(hr):.1f}",
            'F1_mean': np.mean(f1),
            'F1_std': np.std(f1),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('F1_mean', ascending=False)
    summary_df = summary_df.drop(columns=['F1_mean', 'F1_std'])

    print("\nPreference metrics at K=10 (mean ± std over 5 folds):\n")
    print(summary_df.to_string(index=False))

    out_csv = 'cross_validation_5fold_results.csv'
    summary_df.to_csv(out_csv, index=False)
    print(f"\nOK Saved: {out_csv}")

    # Restore original 80/20 split and re-run all main pipeline stages so everything is updated
    print("\n" + "=" * 70)
    print("RESTORING MAIN PIPELINE (80/20 split + all stages)")
    print("=" * 70)

    stages = [
        ('stage1_train_test_split.py', 'Stage 1 (train/test split)'),
        ('stage2_generate_recommendations.py', 'Stage 2 (recommendations)'),
        ('stage3_calculate_portions.py', 'Stage 3 (portions + safety)'),
        ('stage4_evaluate_preference.py', 'Stage 4 (preference evaluation)'),
        ('stage5_combined_evaluation.py', 'Stage 5 (combined safety+preference)'),
        ('stage6_statistical_testing.py', 'Stage 6 (statistical tests)'),
        ('stage7_visualization.py', 'Stage 7 (visualization)'),
    ]
    for script, name in stages:
        if not os.path.exists(script):
            print(f"  Skipping {name} (script not found: {script})")
            continue
        print(f"\nRunning {name}...")
        if run_stage(script, name):
            print(f"  OK {name} complete")
        else:
            print(f"  FAIL: {name} failed (pipeline may be incomplete)")

    print("\n" + "=" * 70)
    print("DONE. 5-fold results: cross_validation_5fold_results.csv")
    print("Main pipeline: train/test and all stage outputs updated to 80/20 split.")
    print("=" * 70)


if __name__ == '__main__':
    main()
