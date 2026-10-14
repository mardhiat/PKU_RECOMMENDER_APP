"""Small rating lookups that do not depend on dataframe index alignment."""

import pandas as pd


def rating_for(ratings_df, user_id, food):
    """Return one user's rating for a food, or None when that pair is absent.

    Filter user and food together. Selecting the food first and then indexing
    with the original frame's labels can return a different user's rating.
    """
    if ratings_df is None or ratings_df.empty:
        return None
    food_key = str(food).lower()
    mask = (ratings_df["user_id"] == user_id) & (
        ratings_df["food"].astype(str).str.lower() == food_key
    )
    rows = ratings_df.loc[mask, "rating"]
    if rows.empty:
        return None
    return float(rows.iloc[0])


def recommendation_universe(rated_foods):
    """Foods the offline evaluation is allowed to recommend.

    Metrics are computed inside the set of foods someone has already rated.
    They are not a cold-start score over the full catalog.
    """
    return set(rated_foods)
