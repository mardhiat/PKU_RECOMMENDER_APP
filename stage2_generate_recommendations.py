import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import random

print("="*70)
print("STAGE 2: GENERATE RECOMMENDATIONS (TWO-TIER: SELECTED + ALL)")
print("="*70)


# ============================================================
# STEP 2.1: LOADING DATA
# ============================================================

print("\nSTEP 2.1: LOADING DATA")

# Check if required files exist
required_files = [
    'data_train_ratings.csv',
    'data_test_ratings.csv',
    'data_test_users.csv',
    'data_food_database.csv',
    'data_meal_ingredients.csv'
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\n❌ ERROR: {file} not found!")
        print("Please run previous stages first.")
        exit()

# Load data
train_df = pd.read_csv('data_train_ratings.csv')
test_df = pd.read_csv('data_test_ratings.csv')
eligible_users_df = pd.read_csv('data_test_users.csv')
food_db_df = pd.read_csv('data_food_database.csv')
meal_ingredients_df = pd.read_csv('data_meal_ingredients.csv')

print(f"\nLoaded data files:")
print(f"  - Train ratings: {len(train_df)} ratings")
print(f"  - Test ratings: {len(test_df)} ratings")
print(f"  - Eligible users: {len(eligible_users_df)} users")
print(f"  - Food database: {len(food_db_df)} foods")
print(f"  - Meal ingredients: {len(meal_ingredients_df)} meals")

# Get list of test users
test_users = test_df['user_name'].unique()
print(f"\nWill generate recommendations for {len(test_users)} users")

# Create food database dictionary for easy lookup
food_db = {}
for _, row in food_db_df.iterrows():
    food_db[row['food_name']] = {
        'phe_mg_per_100g': row['phe_mg_per_100g'],
        'protein_g_per_100g': row['protein_g_per_100g'],
        'energy_kcal_per_100g': row['energy_kcal_per_100g'],
        'serving_size_g': row.get('serving_size_g', 100.0)
    }

print(f"Food database ready with {len(food_db)} foods")


# ============================================================
# STEP 2.2: BUILD INGREDIENT-BASED LOOKUP
# ============================================================

print("\n" + "="*70)
print("🔧 LOADING INGREDIENT & CUISINE DATA")
print("="*70)

# Create meal -> ingredients mapping
meal_to_ingredients = {}
meal_to_cuisine = {}

for _, row in meal_ingredients_df.iterrows():
    meal_name = row['full_name'].lower().strip()
    ingredients = row['ingredients'].split('|')
    ingredients = [ing.lower().strip() for ing in ingredients]
    
    meal_to_ingredients[meal_name] = set(ingredients)
    meal_to_cuisine[meal_name] = row['cuisine']

print(f"✓ Loaded ingredient data for {len(meal_to_ingredients)} meals")
print(f"✓ Loaded cuisine data for {len(meal_to_cuisine)} meals")

# Get all rated foods and check ingredient coverage
all_rated_foods = set(train_df['food'].unique()) | set(test_df['food'].unique())
rated_with_ingredients = sum(1 for food in all_rated_foods 
                            if food.lower().strip() in meal_to_ingredients)

print(f"✓ Ingredient coverage: {rated_with_ingredients}/{len(all_rated_foods)} rated foods ({rated_with_ingredients/len(all_rated_foods)*100:.1f}%)")


# ============================================================
# STEP 2.3: RESTRICT RECOMMENDATION SPACE TO RATED FOODS
# ============================================================

print("\n" + "="*70)
print("⚠️  RESTRICTING RECOMMENDATION SPACE")
print("="*70)

# CRITICAL: Only recommend foods that have been rated by SOMEONE
rated_foods_universe = set(all_rated_foods)

print(f"Total foods in database: {len(food_db)}")
print(f"Foods rated by users: {len(rated_foods_universe)}")
print(f"✓ All algorithms will ONLY recommend from the {len(rated_foods_universe)} rated foods")


# ============================================================
# STEP 2.4: HELPER FUNCTIONS
# ============================================================

def get_ingredient_vector(meal_name):
    """Get ingredient set for a meal"""
    clean_name = meal_name.lower().strip()
    return meal_to_ingredients.get(clean_name, set())


def get_cuisine(meal_name):
    """Get cuisine for a meal"""
    clean_name = meal_name.lower().strip()
    return meal_to_cuisine.get(clean_name, None)


# ============================================================
# STEP 2.5: CONTENT-BASED FILTERING (TWO VERSIONS)
# ============================================================

print("\n" + "="*70)
print("STEP 2.5: IMPLEMENTING CONTENT-BASED FILTERING")
print("="*70)

K = 10  # Top-K recommendations


def content_based_recommendations_selected(user_name, train_df, K=10):
    """
    Recommend foods from USER'S PREFERRED CUISINES
    Based on ingredient similarity to liked foods
    """
    # Get user's training data
    user_train = train_df[train_df['user_name'] == user_name]
    
    # Get foods user liked (rating >= 3)
    liked_foods = user_train[user_train['rating'] >= 3]['food'].tolist()
    
    if not liked_foods:
        return []
    
    # Get ingredient sets for liked foods
    liked_ingredient_sets = []
    liked_cuisines = set()
    
    for food in liked_foods:
        ingredients = get_ingredient_vector(food)
        if ingredients:
            liked_ingredient_sets.append(ingredients)
        
        cuisine = get_cuisine(food)
        if cuisine:
            liked_cuisines.add(cuisine)
    
    if not liked_ingredient_sets:
        return []
    
    if not liked_cuisines:
        return []
    
    # Get foods user has already rated
    rated_foods = set(user_train['food'].str.lower().str.strip())
    
    # FILTER: Only recommend from user's preferred cuisines
    candidate_foods = []
    for food in rated_foods_universe:
        food_lower = food.lower().strip()
        candidate_cuisine = get_cuisine(food)
        
        # CRITICAL: Must be from a cuisine the user has liked
        if (food_lower not in rated_foods and 
            food_lower in meal_to_ingredients and
            candidate_cuisine in liked_cuisines):
            candidate_foods.append(food)
    
    if not candidate_foods:
        return []
    
    # Calculate ingredient similarity for each candidate
    food_scores = {}
    
    for candidate_food in candidate_foods:
        candidate_ingredients = get_ingredient_vector(candidate_food)
        
        if not candidate_ingredients:
            continue
        
        # Calculate Jaccard similarity to all liked foods
        similarities = []
        
        for liked_ingredients in liked_ingredient_sets:
            intersection = len(candidate_ingredients & liked_ingredients)
            union = len(candidate_ingredients | liked_ingredients)
            
            if union > 0:
                jaccard_sim = intersection / union
                similarities.append(jaccard_sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            food_scores[candidate_food] = avg_similarity
    
    # Sort by score and return top K
    sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


def content_based_recommendations_all(user_name, train_df, K=10):
    """
    Recommend foods from ALL CUISINES
    Based on ingredient similarity to liked foods
    """
    # Get user's training data
    user_train = train_df[train_df['user_name'] == user_name]
    
    # Get foods user liked (rating >= 3)
    liked_foods = user_train[user_train['rating'] >= 3]['food'].tolist()
    
    if not liked_foods:
        return []
    
    # Get ingredient sets for liked foods
    liked_ingredient_sets = []
    
    for food in liked_foods:
        ingredients = get_ingredient_vector(food)
        if ingredients:
            liked_ingredient_sets.append(ingredients)
    
    if not liked_ingredient_sets:
        return []
    
    # Get foods user has already rated
    rated_foods = set(user_train['food'].str.lower().str.strip())
    
    # NO CUISINE FILTER: Recommend from any cuisine
    candidate_foods = []
    for food in rated_foods_universe:
        food_lower = food.lower().strip()
        
        if food_lower not in rated_foods and food_lower in meal_to_ingredients:
            candidate_foods.append(food)
    
    if not candidate_foods:
        return []
    
    # Calculate ingredient similarity for each candidate
    food_scores = {}
    
    for candidate_food in candidate_foods:
        candidate_ingredients = get_ingredient_vector(candidate_food)
        
        if not candidate_ingredients:
            continue
        
        # Calculate Jaccard similarity to all liked foods
        similarities = []
        
        for liked_ingredients in liked_ingredient_sets:
            intersection = len(candidate_ingredients & liked_ingredients)
            union = len(candidate_ingredients | liked_ingredients)
            
            if union > 0:
                jaccard_sim = intersection / union
                similarities.append(jaccard_sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            food_scores[candidate_food] = avg_similarity
    
    # Sort by score and return top K
    sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


print("✓ Content-Based (Selected) implemented - recommends from user's cuisines")
print("✓ Content-Based (All) implemented - recommends from any cuisine")


# ============================================================
# STEP 2.6: COLLABORATIVE FILTERING (TWO VERSIONS)
# ============================================================

print("\n" + "="*70)
print("STEP 2.6: IMPLEMENTING COLLABORATIVE FILTERING")
print("="*70)


def collaborative_filtering_recommendations_selected(user_name, train_df, K=10):
    """
    Recommend based on similar users' preferences
    FILTERED to user's preferred cuisines
    """
    # Get user's preferred cuisines
    user_train = train_df[train_df['user_name'] == user_name]
    liked_foods = user_train[user_train['rating'] >= 3]['food'].tolist()
    
    liked_cuisines = set()
    for food in liked_foods:
        cuisine = get_cuisine(food)
        if cuisine:
            liked_cuisines.add(cuisine)
    
    if not liked_cuisines:
        return []
    
    # Build user-item matrix
    user_item_matrix = train_df.pivot_table(
        index='user_name', 
        columns='food', 
        values='rating'
    ).fillna(0)
    
    if user_name not in user_item_matrix.index:
        return []
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    # Find similar users
    similar_users = user_similarity_df[user_name].sort_values(ascending=False)[1:6]
    
    # Get foods rated highly by similar users
    rated_foods = set(user_train['food'].str.lower().str.strip())
    candidate_scores = {}
    
    for similar_user, similarity_score in similar_users.items():
        similar_user_ratings = train_df[
            (train_df['user_name'] == similar_user) & 
            (train_df['rating'] >= 3)
        ]
        
        for _, row in similar_user_ratings.iterrows():
            food = row['food']
            food_lower = food.lower().strip()
            candidate_cuisine = get_cuisine(food)
            
            # FILTER: Only from user's preferred cuisines
            if (food_lower not in rated_foods and 
                food in rated_foods_universe and
                candidate_cuisine in liked_cuisines):
                
                if food not in candidate_scores:
                    candidate_scores[food] = 0
                candidate_scores[food] += similarity_score * row['rating']
    
    # Sort and return top K
    sorted_foods = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


def collaborative_filtering_recommendations_all(user_name, train_df, K=10):
    """
    Recommend based on similar users' preferences
    From ALL cuisines
    """
    # Build user-item matrix
    user_item_matrix = train_df.pivot_table(
        index='user_name', 
        columns='food', 
        values='rating'
    ).fillna(0)
    
    if user_name not in user_item_matrix.index:
        return []
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    # Find similar users
    similar_users = user_similarity_df[user_name].sort_values(ascending=False)[1:6]
    
    # Get foods rated highly by similar users
    user_train = train_df[train_df['user_name'] == user_name]
    rated_foods = set(user_train['food'].str.lower().str.strip())
    candidate_scores = {}
    
    for similar_user, similarity_score in similar_users.items():
        similar_user_ratings = train_df[
            (train_df['user_name'] == similar_user) & 
            (train_df['rating'] >= 3)
        ]
        
        for _, row in similar_user_ratings.iterrows():
            food = row['food']
            food_lower = food.lower().strip()
            
            # NO FILTER: Any cuisine
            if food_lower not in rated_foods and food in rated_foods_universe:
                if food not in candidate_scores:
                    candidate_scores[food] = 0
                candidate_scores[food] += similarity_score * row['rating']
    
    # Sort and return top K
    sorted_foods = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


print("✓ Collaborative (Selected) implemented - recommends from user's cuisines")
print("✓ Collaborative (All) implemented - recommends from any cuisine")


# ============================================================
# STEP 2.7: HYBRID FILTERING (TWO VERSIONS)
# ============================================================

print("\n" + "="*70)
print("STEP 2.7: IMPLEMENTING HYBRID FILTERING")
print("="*70)


def hybrid_recommendations_selected(user_name, train_df, K=10, alpha=0.5):
    """
    Combine content-based and collaborative
    FILTERED to user's preferred cuisines
    """
    cb_recs = content_based_recommendations_selected(user_name, train_df, K=20)
    collab_recs = collaborative_filtering_recommendations_selected(user_name, train_df, K=20)
    
    # Combine scores
    combined_scores = {}
    
    # Normalize and weight content-based scores
    if cb_recs:
        max_cb_score = max(score for _, score in cb_recs)
        for food, score in cb_recs:
            normalized_score = score / max_cb_score if max_cb_score > 0 else 0
            combined_scores[food] = alpha * normalized_score
    
    # Normalize and weight collaborative scores
    if collab_recs:
        max_collab_score = max(score for _, score in collab_recs)
        for food, score in collab_recs:
            normalized_score = score / max_collab_score if max_collab_score > 0 else 0
            if food in combined_scores:
                combined_scores[food] += (1 - alpha) * normalized_score
            else:
                combined_scores[food] = (1 - alpha) * normalized_score
    
    # Sort and return top K
    sorted_foods = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


def hybrid_recommendations_all(user_name, train_df, K=10, alpha=0.5):
    """
    Combine content-based and collaborative
    From ALL cuisines
    """
    cb_recs = content_based_recommendations_all(user_name, train_df, K=20)
    collab_recs = collaborative_filtering_recommendations_all(user_name, train_df, K=20)
    
    # Combine scores
    combined_scores = {}
    
    # Normalize and weight content-based scores
    if cb_recs:
        max_cb_score = max(score for _, score in cb_recs)
        for food, score in cb_recs:
            normalized_score = score / max_cb_score if max_cb_score > 0 else 0
            combined_scores[food] = alpha * normalized_score
    
    # Normalize and weight collaborative scores
    if collab_recs:
        max_collab_score = max(score for _, score in collab_recs)
        for food, score in collab_recs:
            normalized_score = score / max_collab_score if max_collab_score > 0 else 0
            if food in combined_scores:
                combined_scores[food] += (1 - alpha) * normalized_score
            else:
                combined_scores[food] = (1 - alpha) * normalized_score
    
    # Sort and return top K
    sorted_foods = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


print("✓ Hybrid (Selected) implemented - recommends from user's cuisines")
print("✓ Hybrid (All) implemented - recommends from any cuisine")


# ============================================================
# STEP 2.8: BASELINE ALGORITHMS (TWO VERSIONS)
# ============================================================

print("\n" + "="*70)
print("STEP 2.8: IMPLEMENTING BASELINE ALGORITHMS")
print("="*70)


def popularity_recommendations_selected(user_name, train_df, K=10):
    """
    Recommend most popular foods
    FILTERED to user's preferred cuisines
    """
    # Get user's preferred cuisines
    user_train = train_df[train_df['user_name'] == user_name]
    liked_foods = user_train[user_train['rating'] >= 3]['food'].tolist()
    
    liked_cuisines = set()
    for food in liked_foods:
        cuisine = get_cuisine(food)
        if cuisine:
            liked_cuisines.add(cuisine)
    
    if not liked_cuisines:
        return []
    
    # Calculate popularity (average rating)
    food_popularity = train_df.groupby('food')['rating'].agg(['mean', 'count'])
    
    # Filter to foods not rated by user
    rated_foods = set(user_train['food'].str.lower().str.strip())
    
    popular_foods = []
    for food, stats in food_popularity.iterrows():
        food_lower = food.lower().strip()
        candidate_cuisine = get_cuisine(food)
        
        # FILTER: Only from user's preferred cuisines
        if (food_lower not in rated_foods and 
            food in rated_foods_universe and
            stats['count'] >= 2 and
            candidate_cuisine in liked_cuisines):
            popular_foods.append((food, stats['mean']))
    
    # Sort by rating and return top K
    sorted_foods = sorted(popular_foods, key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


def popularity_recommendations_all(user_name, train_df, K=10):
    """
    Recommend most popular foods
    From ALL cuisines
    """
    user_train = train_df[train_df['user_name'] == user_name]
    rated_foods = set(user_train['food'].str.lower().str.strip())
    
    # Calculate popularity (average rating)
    food_popularity = train_df.groupby('food')['rating'].agg(['mean', 'count'])
    
    popular_foods = []
    for food, stats in food_popularity.iterrows():
        food_lower = food.lower().strip()
        
        # NO FILTER: Any cuisine
        if (food_lower not in rated_foods and 
            food in rated_foods_universe and
            stats['count'] >= 2):
            popular_foods.append((food, stats['mean']))
    
    # Sort by rating and return top K
    sorted_foods = sorted(popular_foods, key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


def random_recommendations(user_name, train_df, K=10):
    """Random baseline (no cuisine filtering needed)"""
    user_train = train_df[train_df['user_name'] == user_name]
    rated_foods = set(user_train['food'].str.lower().str.strip())
    
    # Get unrated foods
    unrated_foods = [food for food in rated_foods_universe 
                     if food.lower().strip() not in rated_foods]
    
    # Random sample
    sample_size = min(K, len(unrated_foods))
    if sample_size == 0:
        return []
    
    sampled = random.sample(unrated_foods, sample_size)
    return [(food, 1.0) for food in sampled]


print("✓ Popularity (Selected) implemented")
print("✓ Popularity (All) implemented")
print("✓ Random baseline implemented")


# ============================================================
# STEP 2.9: GENERATE RECOMMENDATIONS FOR ALL TEST USERS
# ============================================================

print("\n" + "="*70)
print("STEP 2.9: GENERATING RECOMMENDATIONS FOR ALL TEST USERS")
print("="*70)

all_recommendations = {
    'content_based_selected': {},
    'content_based_all': {},
    'collaborative_selected': {},
    'collaborative_all': {},
    'hybrid_selected': {},
    'hybrid_all': {},
    'popularity_selected': {},
    'popularity_all': {},
    'random': {}
}

for i, user_name in enumerate(test_users, 1):
    if i % 5 == 0:
        print(f"Processing user {i}/{len(test_users)}...")
    
    # Content-Based
    all_recommendations['content_based_selected'][user_name] = \
        content_based_recommendations_selected(user_name, train_df, K=K)
    all_recommendations['content_based_all'][user_name] = \
        content_based_recommendations_all(user_name, train_df, K=K)
    
    # Collaborative
    all_recommendations['collaborative_selected'][user_name] = \
        collaborative_filtering_recommendations_selected(user_name, train_df, K=K)
    all_recommendations['collaborative_all'][user_name] = \
        collaborative_filtering_recommendations_all(user_name, train_df, K=K)
    
    # Hybrid
    all_recommendations['hybrid_selected'][user_name] = \
        hybrid_recommendations_selected(user_name, train_df, K=K)
    all_recommendations['hybrid_all'][user_name] = \
        hybrid_recommendations_all(user_name, train_df, K=K)
    
    # Popularity
    all_recommendations['popularity_selected'][user_name] = \
        popularity_recommendations_selected(user_name, train_df, K=K)
    all_recommendations['popularity_all'][user_name] = \
        popularity_recommendations_all(user_name, train_df, K=K)
    
    # Random
    all_recommendations['random'][user_name] = \
        random_recommendations(user_name, train_df, K=K)

print(f"\n✓ Generated recommendations for {len(test_users)} users")


# ============================================================
# STEP 2.10: SUMMARY STATISTICS
# ============================================================

print("\n" + "="*70)
print("RECOMMENDATION STATISTICS")
print("="*70)

for algo_name, recs in all_recommendations.items():
    total_recs = sum(len(user_recs) for user_recs in recs.values())
    avg_recs = total_recs / len(test_users) if len(test_users) > 0 else 0
    
    print(f"\n{algo_name.upper()}:")
    print(f"  Total recommendations: {total_recs}")
    print(f"  Average per user: {avg_recs:.1f}/{K}")
    
    # Count how many users got recommendations
    users_with_recs = sum(1 for user_recs in recs.values() if len(user_recs) > 0)
    print(f"  Users with recommendations: {users_with_recs}/{len(test_users)}")


# ============================================================
# STEP 2.11: SAVE RESULTS
# ============================================================

print("\n" + "="*70)
print("SAVING RECOMMENDATIONS")
print("="*70)

# Save to pickle
with open('recommendations_all_algorithms.pkl', 'wb') as f:
    pickle.dump(all_recommendations, f)

print("✓ Saved: recommendations_all_algorithms.pkl")

# Create summary DataFrame
summary_data = []
for algo_name, user_recs in all_recommendations.items():
    for user_name, recs in user_recs.items():
        for rank, (food, score) in enumerate(recs, 1):
            summary_data.append({
                'algorithm': algo_name,
                'user_name': user_name,
                'food': food,
                'rank': rank,
                'score': score
            })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('recommendations_summary.csv', index=False)
print("✓ Saved: recommendations_summary.csv")

print("\n" + "="*70)
print("STAGE 2 COMPLETE")
print("="*70)
print(f"""
Generated {len(all_recommendations)} recommendation variants:
  1. Content-Based (Selected) - {sum(len(r) for r in all_recommendations['content_based_selected'].values())} recs
  2. Content-Based (All) - {sum(len(r) for r in all_recommendations['content_based_all'].values())} recs
  3. Collaborative (Selected) - {sum(len(r) for r in all_recommendations['collaborative_selected'].values())} recs
  4. Collaborative (All) - {sum(len(r) for r in all_recommendations['collaborative_all'].values())} recs
  5. Hybrid (Selected) - {sum(len(r) for r in all_recommendations['hybrid_selected'].values())} recs
  6. Hybrid (All) - {sum(len(r) for r in all_recommendations['hybrid_all'].values())} recs
  7. Popularity (Selected) - {sum(len(r) for r in all_recommendations['popularity_selected'].values())} recs
  8. Popularity (All) - {sum(len(r) for r in all_recommendations['popularity_all'].values())} recs
  9. Random - {sum(len(r) for r in all_recommendations['random'].values())} recs

Next: Run Stage 3 to calculate safety constraints
""")