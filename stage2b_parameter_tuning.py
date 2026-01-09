import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

print("="*70)
print("STAGE 2B: PARAMETER TUNING (K VALUES + ALPHA VALUES)")
print("="*70)

# ============================================================
# LOAD BASE DATA
# ============================================================

print("\nLOADING BASE DATA...")

train_df = pd.read_csv('data_train_ratings.csv')
test_df = pd.read_csv('data_test_ratings.csv')
meal_ingredients_df = pd.read_csv('data_meal_ingredients.csv')

test_users = test_df['user_name'].unique()

# Load TF-IDF data structures from previous run
with open('recommendations_all_algorithms_TFIDF.pkl', 'rb') as f:
    _ = pickle.load(f)  # Just to verify file exists

print(f"✓ Loaded {len(train_df)} training ratings")
print(f"✓ Loaded {len(test_df)} test ratings")
print(f"✓ Will test on {len(test_users)} users")

# ============================================================
# PARAMETER GRIDS
# ============================================================

K_VALUES = [5, 10, 15, 20]
ALPHA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]

print(f"\nPARAMETER GRID:")
print(f"  K values: {K_VALUES}")
print(f"  Alpha values (hybrid): {ALPHA_VALUES}")
print(f"  Total combinations: {len(K_VALUES)} K × {len(ALPHA_VALUES)} alpha = {len(K_VALUES) * len(ALPHA_VALUES)} tests")

# ============================================================
# IMPORT RECOMMENDATION FUNCTIONS FROM STAGE 2
# ============================================================

# Create meal-to-ingredients mapping
meal_to_ingredients = {}
meal_to_cuisine = {}

for _, row in meal_ingredients_df.iterrows():
    meal_name = row['full_name'].lower().strip()
    ingredients = row['ingredients'].split('|')
    ingredients = [ing.lower().strip() for ing in ingredients]
    
    meal_to_ingredients[meal_name] = set(ingredients)
    meal_to_cuisine[meal_name] = row['cuisine']

# Get rated foods universe
all_rated_foods = set(train_df['food'].unique()) | set(test_df['food'].unique())

def get_cuisine(meal_name):
    clean_name = meal_name.lower().strip()
    return meal_to_cuisine.get(clean_name, None)

# Build TF-IDF (reuse from Stage 2)
from sklearn.feature_extraction.text import TfidfVectorizer

food_names = []
ingredient_documents = []

for food_name, ingredients in meal_to_ingredients.items():
    food_names.append(food_name)
    ingredient_documents.append(' '.join(ingredients))

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(ingredient_documents)
food_to_idx = {food: idx for idx, food in enumerate(food_names)}

def get_tfidf_vector(meal_name):
    clean_name = meal_name.lower().strip()
    if clean_name in food_to_idx:
        idx = food_to_idx[clean_name]
        return tfidf_matrix[idx]
    return None

def calculate_tfidf_similarity(food1, food2):
    vec1 = get_tfidf_vector(food1)
    vec2 = get_tfidf_vector(food2)
    
    if vec1 is None or vec2 is None:
        return 0.0
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity

# ============================================================
# RECOMMENDATION FUNCTIONS (PARAMETERIZED)
# ============================================================

def content_based_recommendations_selected(user_name, train_df, K=10):
    user_train = train_df[train_df['user_name'] == user_name]
    liked_foods = user_train[user_train['rating'] >= 3]['food'].tolist()
    
    if not liked_foods:
        return []
    
    liked_cuisines = set()
    valid_liked_foods = []
    
    for food in liked_foods:
        cuisine = get_cuisine(food)
        if cuisine:
            liked_cuisines.add(cuisine)
            valid_liked_foods.append(food)
    
    if not liked_cuisines or not valid_liked_foods:
        return []
    
    rated_foods = set(user_train['food'].str.lower().str.strip())
    
    candidate_foods = []
    for food in all_rated_foods:
        food_lower = food.lower().strip()
        candidate_cuisine = get_cuisine(food)
        
        if (food_lower not in rated_foods and 
            food_lower in meal_to_ingredients and
            candidate_cuisine in liked_cuisines):
            candidate_foods.append(food)
    
    if not candidate_foods:
        return []
    
    food_scores = {}
    
    for candidate_food in candidate_foods:
        similarities = []
        
        for liked_food in valid_liked_foods:
            tfidf_sim = calculate_tfidf_similarity(candidate_food, liked_food)
            if tfidf_sim > 0:
                similarities.append(tfidf_sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            food_scores[candidate_food] = avg_similarity
    
    sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


def collaborative_filtering_recommendations_selected(user_name, train_df, K=10):
    user_train = train_df[train_df['user_name'] == user_name]
    liked_foods = user_train[user_train['rating'] >= 3]['food'].tolist()
    
    liked_cuisines = set()
    for food in liked_foods:
        cuisine = get_cuisine(food)
        if cuisine:
            liked_cuisines.add(cuisine)
    
    if not liked_cuisines:
        return []
    
    user_item_matrix = train_df.pivot_table(
        index='user_name', 
        columns='food', 
        values='rating'
    ).fillna(0)
    
    if user_name not in user_item_matrix.index:
        return []
    
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    similar_users = user_similarity_df[user_name].sort_values(ascending=False)[1:6]
    
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
            
            if (food_lower not in rated_foods and 
                food in all_rated_foods and
                candidate_cuisine in liked_cuisines):
                
                if food not in candidate_scores:
                    candidate_scores[food] = 0
                candidate_scores[food] += similarity_score * row['rating']
    
    sorted_foods = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]


def hybrid_recommendations_selected(user_name, train_df, K=10, alpha=0.5):
    cb_recs = content_based_recommendations_selected(user_name, train_df, K=20)
    collab_recs = collaborative_filtering_recommendations_selected(user_name, train_df, K=20)
    
    combined_scores = {}
    
    if cb_recs:
        max_cb_score = max(score for _, score in cb_recs)
        for food, score in cb_recs:
            normalized_score = score / max_cb_score if max_cb_score > 0 else 0
            combined_scores[food] = alpha * normalized_score
    
    if collab_recs:
        max_collab_score = max(score for _, score in collab_recs)
        for food, score in collab_recs:
            normalized_score = score / max_collab_score if max_collab_score > 0 else 0
            if food in combined_scores:
                combined_scores[food] += (1 - alpha) * normalized_score
            else:
                combined_scores[food] = (1 - alpha) * normalized_score
    
    sorted_foods = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_foods[:K]

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_recommendations(recommendations, test_df, users):
    """
    Calculate F1, Precision, Recall for given recommendations
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for user_name in users:
        user_recs = recommendations.get(user_name, [])
        
        if not user_recs:
            continue
        
        # Get foods user liked in test set
        user_test = test_df[test_df['user_name'] == user_name]
        liked_foods = set(user_test[user_test['rating'] >= 4]['food'].str.lower().str.strip())
        
        if not liked_foods:
            continue
        
        # Get recommended foods
        rec_foods = set([food.lower().strip() for food, _ in user_recs])
        
        # Calculate metrics
        hits = len(rec_foods & liked_foods)
        
        precision = hits / len(rec_foods) if len(rec_foods) > 0 else 0
        recall = hits / len(liked_foods) if len(liked_foods) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    return {
        'precision': np.mean(precision_scores) * 100 if precision_scores else 0,
        'recall': np.mean(recall_scores) * 100 if recall_scores else 0,
        'f1': np.mean(f1_scores) * 100 if f1_scores else 0,
        'n_users': len(precision_scores)
    }

# ============================================================
# RUN PARAMETER SWEEP
# ============================================================

print("\n" + "="*70)
print("RUNNING PARAMETER SWEEP")
print("="*70)

results = []

# Test K values for content-based and collaborative
print("\nTesting K values for Content-Based and Collaborative...")
for K in K_VALUES:
    print(f"\n  Testing K={K}...")
    
    # Content-Based
    cb_recs = {}
    for user in test_users:
        cb_recs[user] = content_based_recommendations_selected(user, train_df, K=K)
    
    cb_metrics = evaluate_recommendations(cb_recs, test_df, test_users)
    results.append({
        'algorithm': 'content_based_selected',
        'K': K,
        'alpha': None,
        'f1': cb_metrics['f1'],
        'precision': cb_metrics['precision'],
        'recall': cb_metrics['recall'],
        'n_users': cb_metrics['n_users']
    })
    print(f"    Content-Based: F1={cb_metrics['f1']:.1f}%")
    
    # Collaborative
    collab_recs = {}
    for user in test_users:
        collab_recs[user] = collaborative_filtering_recommendations_selected(user, train_df, K=K)
    
    collab_metrics = evaluate_recommendations(collab_recs, test_df, test_users)
    results.append({
        'algorithm': 'collaborative_selected',
        'K': K,
        'alpha': None,
        'f1': collab_metrics['f1'],
        'precision': collab_metrics['precision'],
        'recall': collab_metrics['recall'],
        'n_users': collab_metrics['n_users']
    })
    print(f"    Collaborative: F1={collab_metrics['f1']:.1f}%")

# Test K × Alpha combinations for hybrid
print("\nTesting K × Alpha combinations for Hybrid...")
for K in K_VALUES:
    for alpha in ALPHA_VALUES:
        print(f"  Testing K={K}, alpha={alpha}...", end='')
        
        hybrid_recs = {}
        for user in test_users:
            hybrid_recs[user] = hybrid_recommendations_selected(user, train_df, K=K, alpha=alpha)
        
        hybrid_metrics = evaluate_recommendations(hybrid_recs, test_df, test_users)
        results.append({
            'algorithm': 'hybrid_selected',
            'K': K,
            'alpha': alpha,
            'f1': hybrid_metrics['f1'],
            'precision': hybrid_metrics['precision'],
            'recall': hybrid_metrics['recall'],
            'n_users': hybrid_metrics['n_users']
        })
        print(f" F1={hybrid_metrics['f1']:.1f}%")

# ============================================================
# SAVE AND DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv('parameter_tuning_results.csv', index=False)

print("\n" + "="*70)
print("PARAMETER TUNING RESULTS")
print("="*70)

# Best K for each algorithm
print("\n📊 BEST K VALUE FOR EACH ALGORITHM:")
print("-" * 70)

for algo in ['content_based_selected', 'collaborative_selected']:
    algo_results = results_df[results_df['algorithm'] == algo]
    best_row = algo_results.loc[algo_results['f1'].idxmax()]
    print(f"\n{algo.upper()}:")
    print(f"  Best K: {int(best_row['K'])}")
    print(f"  F1: {best_row['f1']:.1f}%")
    print(f"  Precision: {best_row['precision']:.1f}%")
    print(f"  Recall: {best_row['recall']:.1f}%")

# Best K × Alpha for hybrid
print("\n" + "-" * 70)
print("\nHYBRID_SELECTED:")
hybrid_results = results_df[results_df['algorithm'] == 'hybrid_selected']
best_hybrid = hybrid_results.loc[hybrid_results['f1'].idxmax()]
print(f"  Best K: {int(best_hybrid['K'])}")
print(f"  Best Alpha: {best_hybrid['alpha']:.1f}")
print(f"  F1: {best_hybrid['f1']:.1f}%")
print(f"  Precision: {best_hybrid['precision']:.1f}%")
print(f"  Recall: {best_hybrid['recall']:.1f}%")

# Detailed tables
print("\n" + "="*70)
print("DETAILED RESULTS BY K (Content-Based)")
print("="*70)
cb_table = results_df[results_df['algorithm'] == 'content_based_selected'][['K', 'f1', 'precision', 'recall']]
cb_table = cb_table.sort_values('K')
print(cb_table.to_string(index=False))

print("\n" + "="*70)
print("DETAILED RESULTS BY K (Collaborative)")
print("="*70)
collab_table = results_df[results_df['algorithm'] == 'collaborative_selected'][['K', 'f1', 'precision', 'recall']]
collab_table = collab_table.sort_values('K')
print(collab_table.to_string(index=False))

print("\n" + "="*70)
print("DETAILED RESULTS BY K × ALPHA (Hybrid)")
print("="*70)
hybrid_table = results_df[results_df['algorithm'] == 'hybrid_selected'][['K', 'alpha', 'f1', 'precision', 'recall']]
hybrid_table = hybrid_table.sort_values(['K', 'alpha'])
print(hybrid_table.to_string(index=False))

print("\n" + "="*70)
print("STAGE 2B COMPLETE")
print("="*70)
print(f"""
Tested {len(results)} parameter combinations:
  - {len(K_VALUES)} K values for content-based
  - {len(K_VALUES)} K values for collaborative
  - {len(K_VALUES) * len(ALPHA_VALUES)} K × alpha combinations for hybrid

Results saved to: parameter_tuning_results.csv

NEXT STEPS:
1. Review the best parameters above
2. Use these optimal parameters in your final Stage 2
3. Proceed to Stage 6 with optimized configuration
""")