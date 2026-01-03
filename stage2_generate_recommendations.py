import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import random

print("STAGE 2: GENERATE RECOMMENDATIONS")


# STEP 2.1: LOAD DATA FROM PREVIOUS STAGES


print("STEP 2.1: LOADING DATA")

# Check if required files exist
required_files = [
    'train_ratings.csv',
    'test_ratings.csv',
    'eligible_users.csv',
    'data_food_database.csv'
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\n❌ ERROR: {file} not found!")
        print("Please run previous stages first.")
        exit()

# Load data
train_df = pd.read_csv('train_ratings.csv')
test_df = pd.read_csv('test_ratings.csv')
eligible_users_df = pd.read_csv('eligible_users.csv')
food_db_df = pd.read_csv('data_food_database.csv')

print(f"\nLoaded data files:")
print(f"  - Train ratings: {len(train_df)} ratings")
print(f"  - Test ratings: {len(test_df)} ratings")
print(f"  - Eligible users: {len(eligible_users_df)} users")
print(f"  - Food database: {len(food_db_df)} foods")

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


# STEP 2.2: IMPLEMENT RECOMMENDATION ALGORITHMS


print("STEP 2.2: IMPLEMENTING RECOMMENDATION ALGORITHMS")

K = 10  # Top-K recommendations


# ALGORITHM 1: CONTENT-BASED FILTERING


def content_based_recommendations(user_name, train_df, food_db, K=10):
    """
    Recommend foods similar to what user liked
    Similarity based on nutritional profile (PHE, protein, energy)
    """
    # Get user's training data
    user_train = train_df[train_df['user_name'] == user_name]
    
    # Get foods user liked (rating >= 4)
    liked_foods = user_train[user_train['rating'] >= 4]['food'].tolist()
    
    if not liked_foods:
        return []  # User has no liked foods
    
    # Get foods user has already rated (to exclude from recommendations)
    rated_foods = set(user_train['food'].str.lower().str.split('(').str[0].str.strip())
    
    # Get all candidate foods (foods in database that user hasn't rated)
    all_foods = set(food_db.keys())
    candidate_foods = all_foods - rated_foods
    
    if not candidate_foods:
        return []  # No unrated foods
    
    # Calculate similarity scores for each candidate
    food_scores = {}
    
    for candidate_food in candidate_foods:
        candidate_nutrients = food_db.get(candidate_food)
        if candidate_nutrients is None:
            continue
        
        # Calculate average similarity to all liked foods
        similarities = []
        
        for liked_food in liked_foods:
            # Clean liked food name (remove cuisine tag)
            liked_clean = liked_food.lower().split('(')[0].strip()
            liked_nutrients = food_db.get(liked_clean)
            
            if liked_nutrients is None:
                continue
            
            # Calculate nutritional similarity
            # Based on differences in PHE, protein, and energy
            phe_diff = abs(candidate_nutrients['phe_mg_per_100g'] - 
                          liked_nutrients['phe_mg_per_100g'])
            protein_diff = abs(candidate_nutrients['protein_g_per_100g'] - 
                              liked_nutrients['protein_g_per_100g'])
            energy_diff = abs(candidate_nutrients['energy_kcal_per_100g'] - 
                             liked_nutrients['energy_kcal_per_100g'])
            
            # Convert differences to similarity scores (closer = more similar)
            # Using exponential decay: similarity = e^(-difference)
            phe_sim = np.exp(-phe_diff / 100.0)  # Normalize by 100
            protein_sim = np.exp(-protein_diff / 10.0)  # Normalize by 10
            energy_sim = np.exp(-energy_diff / 100.0)  # Normalize by 100
            
            # Average similarity across all nutrients
            similarity = (phe_sim + protein_sim + energy_sim) / 3.0
            similarities.append(similarity)
        
        if similarities:
            # Average similarity to all liked foods
            food_scores[candidate_food] = np.mean(similarities)
    
    # Sort by score and return top K
    sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]

print("Content-Based algorithm implemented")


# ALGORITHM 2: COLLABORATIVE FILTERING


def collaborative_filtering_recommendations(user_name, train_df, K=10):
    """
    Recommend based on similar users' preferences
    Find users with similar taste, recommend what they liked
    """
    # Create user-item matrix
    user_item_matrix = train_df.pivot_table(
        index='user_name',
        columns='food',
        values='rating',
        fill_value=0
    )
    
    # Check if user exists
    if user_name not in user_item_matrix.index:
        return []
    
    # Get target user's ratings vector
    target_user_ratings = user_item_matrix.loc[user_name].values.reshape(1, -1)
    
    # Calculate similarity to all other users using cosine similarity
    similarities = cosine_similarity(target_user_ratings, user_item_matrix.values)[0]
    
    # Get indices of most similar users (excluding self)
    user_indices = np.argsort(similarities)[::-1]
    
    # Remove self from similar users
    similar_user_indices = [idx for idx in user_indices if user_item_matrix.index[idx] != user_name][:5]
    
    # Get foods target user hasn't rated
    target_user_rated = set(train_df[train_df['user_name'] == user_name]['food'])
    
    # Predict ratings for unrated foods based on similar users
    food_scores = {}
    
    for food in user_item_matrix.columns:
        if food not in target_user_rated:
            # Weighted average of similar users' ratings
            weighted_sum = 0
            similarity_sum = 0
            
            for user_idx in similar_user_indices:
                other_user_rating = user_item_matrix.iloc[user_idx][food]
                if other_user_rating > 0:  # User rated this food
                    weighted_sum += similarities[user_idx] * other_user_rating
                    similarity_sum += similarities[user_idx]
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                food_scores[food] = predicted_rating
    
    # Sort by predicted rating and return top K
    sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]

print("Collaborative Filtering algorithm implemented")


# ALGORITHM 3: HYBRID (Content + Collaborative)


def hybrid_recommendations(user_name, train_df, food_db, K=10, alpha=0.6):
    """
    Combine content-based and collaborative filtering
    alpha: weight for content-based (1-alpha for collaborative)
    """
    # Get recommendations from both algorithms
    content_recs = content_based_recommendations(user_name, train_df, food_db, K=20)
    collab_recs = collaborative_filtering_recommendations(user_name, train_df, K=20)
    
    # Convert to dictionaries
    content_scores = {food: score for food, score in content_recs}
    collab_scores = {food: score for food, score in collab_recs}
    
    # Normalize scores to 0-1 range
    def normalize_scores(scores_dict):
        if not scores_dict:
            return {}
        values = list(scores_dict.values())
        max_score = max(values)
        min_score = min(values)
        if max_score == min_score:
            return {k: 1.0 for k in scores_dict}
        return {k: (v - min_score) / (max_score - min_score) 
                for k, v in scores_dict.items()}
    
    content_norm = normalize_scores(content_scores)
    collab_norm = normalize_scores(collab_scores)
    
    # Combine scores using weighted average
    all_foods = set(content_norm.keys()) | set(collab_norm.keys())
    hybrid_scores = {}
    
    for food in all_foods:
        content_score = content_norm.get(food, 0)
        collab_score = collab_norm.get(food, 0)
        hybrid_scores[food] = alpha * content_score + (1 - alpha) * collab_score
    
    # Sort and return top K
    sorted_foods = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]

print("Hybrid algorithm implemented")


# ALGORITHM 4: RANDOM BASELINE


def random_recommendations(user_name, train_df, food_db, K=10, seed=42):
    """
    Random baseline: recommend K random unrated foods
    """
    random.seed(seed)
    
    # Get foods user has rated
    user_rated = set(train_df[train_df['user_name'] == user_name]['food'].str.lower())
    
    # Get all possible foods
    all_foods = list(food_db.keys())
    
    # Filter to unrated foods
    unrated = [f for f in all_foods if f not in user_rated]
    
    if not unrated:
        return []
    
    # Random sample
    if len(unrated) < K:
        selected = unrated
    else:
        selected = random.sample(unrated, K)
    
    # Return with uniform scores
    return [(food, 1.0) for food in selected]

print("Random baseline implemented")


# ALGORITHM 5: POPULARITY BASELINE


def popularity_recommendations(user_name, train_df, K=10):
    """
    Popularity baseline: recommend most commonly liked foods
    """
    # Count how many users liked each food (rating >= 4)
    food_popularity = train_df[train_df['rating'] >= 4].groupby('food').size()
    food_popularity = food_popularity.sort_values(ascending=False)
    
    # Get foods target user hasn't rated
    user_rated = set(train_df[train_df['user_name'] == user_name]['food'])
    
    # Filter to unrated foods
    popular_unrated = [
        (food, count) for food, count in food_popularity.items()
        if food not in user_rated
    ]
    
    return popular_unrated[:K]

print("Popularity baseline implemented")


# STEP 2.3: GENERATE RECOMMENDATIONS FOR ALL TEST USERS


print("STEP 2.3: GENERATING RECOMMENDATIONS FOR ALL TEST USERS")

# Store recommendations for each algorithm
recommendations_dict = {
    'content_based': {},
    'collaborative': {},
    'hybrid': {},
    'random': {},
    'popularity': {}
}

print(f"\nGenerating top-{K} recommendations for {len(test_users)} users...")
print("This may take 1-2 minutes...\n")

for i, user_name in enumerate(test_users, 1):
    if i % 5 == 0 or i == len(test_users):
        print(f"  Progress: {i}/{len(test_users)} users completed...")
    
    # Content-based
    try:
        recommendations_dict['content_based'][user_name] = \
            content_based_recommendations(user_name, train_df, food_db, K=K)
    except Exception as e:
        print(f"  ⚠ Content-based failed for {user_name}: {e}")
        recommendations_dict['content_based'][user_name] = []
    
    # Collaborative
    try:
        recommendations_dict['collaborative'][user_name] = \
            collaborative_filtering_recommendations(user_name, train_df, K=K)
    except Exception as e:
        print(f"  ⚠ Collaborative failed for {user_name}: {e}")
        recommendations_dict['collaborative'][user_name] = []
    
    # Hybrid
    try:
        recommendations_dict['hybrid'][user_name] = \
            hybrid_recommendations(user_name, train_df, food_db, K=K)
    except Exception as e:
        print(f"  ⚠ Hybrid failed for {user_name}: {e}")
        recommendations_dict['hybrid'][user_name] = []
    
    # Random baseline
    try:
        recommendations_dict['random'][user_name] = \
            random_recommendations(user_name, train_df, food_db, K=K, seed=42+i)
    except Exception as e:
        print(f"  ⚠ Random failed for {user_name}: {e}")
        recommendations_dict['random'][user_name] = []
    
    # Popularity baseline
    try:
        recommendations_dict['popularity'][user_name] = \
            popularity_recommendations(user_name, train_df, K=K)
    except Exception as e:
        print(f"  ⚠ Popularity failed for {user_name}: {e}")
        recommendations_dict['popularity'][user_name] = []

print("\nRecommendations generated for all users!")


# STEP 2.4: SAVE RECOMMENDATIONS


print("STEP 2.4: SAVING RECOMMENDATIONS")

# Save to pickle file
with open('recommendations_all_algorithms.pkl', 'wb') as f:
    pickle.dump(recommendations_dict, f)

print(f"\nSaved: recommendations_all_algorithms.pkl")


# STEP 2.5: SUMMARY STATISTICS


print("STEP 2.5: SUMMARY STATISTICS")

print(f"\nRecommendations generated:")
for algorithm in recommendations_dict:
    total_recs = sum(len(recs) for recs in recommendations_dict[algorithm].values())
    users_with_recs = sum(1 for recs in recommendations_dict[algorithm].values() if len(recs) > 0)
    avg_recs = total_recs / len(test_users) if len(test_users) > 0 else 0
    
    print(f"\n{algorithm.upper()}:")
    print(f"  Total recommendations: {total_recs}")
    print(f"  Users with recommendations: {users_with_recs}/{len(test_users)}")
    print(f"  Average recommendations per user: {avg_recs:.1f}")

# Show sample recommendations for first user
sample_user = test_users[0]
print(f"Sample recommendations for user: {sample_user}")

for algorithm in recommendations_dict:
    recs = recommendations_dict[algorithm][sample_user]
    print(f"\n{algorithm.upper()} (top 5):")
    if recs:
        for i, (food, score) in enumerate(recs[:5], 1):
            print(f"  {i}. {food[:50]}... (score: {score:.3f})")
    else:
        print("  No recommendations generated")


# FINAL SUMMARY


print("STAGE 2 COMPLETE - SUMMARY")

print(f"""
Files Created:
  1. recommendations_all_algorithms.pkl - All recommendations for all users

Recommendations Generated:
  • Number of users: {len(test_users)}
  • Recommendations per user: {K}
  • Algorithms: 5 (content-based, collaborative, hybrid, random, popularity)
  • Total recommendations: {len(test_users) * K * 5}

Next Steps:
  Stage 3: Calculate Safe Portions
  - For each recommendation, calculate maximum safe portion
  - Based on user's PHE and protein limits
  - Check if portion is practical (≥30g)
""")

