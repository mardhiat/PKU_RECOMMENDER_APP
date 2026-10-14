"""Repository paths. Pipeline scripts and the app should use these instead of the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CUISINES = DATA / "cuisines"
NUTRITION = DATA / "nutrition"
PROCESSED = DATA / "processed"
FIXTURES = DATA / "fixtures"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
RESULTS = DOCS / "results"


class Paths:
    root = ROOT
    cuisines = CUISINES
    nutrition_dir = NUTRITION
    nutrition_db = NUTRITION / "Nutritional_Data.csv"
    chat_ingredients = NUTRITION / "chat_ingredients.csv"
    baby_foods = NUTRITION / "baby_foods.csv"
    fixture_ratings = FIXTURES / "synthetic_ratings.csv"
    food_db = PROCESSED / "food_database.csv"
    meal_ingredients = PROCESSED / "meal_ingredients.csv"
    meal_clusters = PROCESSED / "meal_clusters.csv"
    cluster_summary = PROCESSED / "cluster_summary.csv"
    missing_foods = PROCESSED / "missing_foods.csv"
    missing_ingredients = PROCESSED / "missing_ingredients.csv"
    ingredient_db = PROCESSED / "ingredient_database.csv"
    user_ratings = PROCESSED / "user_food_ratings.csv"
    user_limits = PROCESSED / "user_limits.csv"
    train_ratings = PROCESSED / "train_ratings.csv"
    test_ratings = PROCESSED / "test_ratings.csv"
    test_users = PROCESSED / "test_users.csv"
    usda_cache = PROCESSED / "usda_cache.json"
    recommendations_pkl = PROCESSED / "recommendations.pkl"
    portions_pkl = PROCESSED / "recommendations_with_portions.pkl"
    preference_pkl = PROCESSED / "preference_results.pkl"
    combined_pkl = PROCESSED / "combined_results.pkl"
    preference_summary = RESULTS / "preference_summary.csv"
    liked_and_safe = RESULTS / "liked_and_safe.csv"
    coverage = RESULTS / "coverage.csv"
    acceptance = RESULTS / "acceptance.csv"
    revised_evaluation = RESULTS / "revised_evaluation.csv"
    stats_preference = RESULTS / "stats_preference.csv"
    stats_combined = RESULTS / "stats_combined.csv"
    summary_txt = RESULTS / "summary.txt"
    tuning_results = RESULTS / "parameter_tuning.csv"
    cv_results = RESULTS / "cross_validation.csv"
    cv_table = RESULTS / "cross_validation_table.csv"
    figure_preference = FIGURES / "preference_metrics.png"
    figure_liked_safe = FIGURES / "liked_and_safe.png"
    figure_cuisine = FIGURES / "cuisine_filtering.png"
    figure_perspectives = FIGURES / "three_perspectives.png"
    figure_summary = FIGURES / "summary_table.png"
    figure_elbow = FIGURES / "meal_clustering_elbow.png"
    figure_clusters = FIGURES / "meal_clustering_analysis.png"
    evaluation_guide = DOCS / "evaluation.md"
    secrets_file = ROOT / ".streamlit" / "secrets.toml"


paths = Paths()
