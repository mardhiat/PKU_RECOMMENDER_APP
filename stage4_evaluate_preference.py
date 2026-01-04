import pandas as pd
import numpy as np
import pickle
import os

print("=" * 70)
print("STAGE 4: EVALUATE PREFERENCE ALIGNMENT (FIXED)")
print("=" * 70)

# STEP 4.1: LOAD DATA

print("\nSTEP 4.1: LOADING DATA")

required_files = [
    'recommendations_with_portions.pkl',
    'test_ratings.csv',
    'train_ratings.csv'
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\n❌ ERROR: {file} not found!")
        exit()

with open('recommendations_with_portions.pkl', 'rb') as f:
    recs_with_portions = pickle.load(f)

test_df = pd.read_csv('test_ratings.csv')
train_df = pd.read_csv('train_ratings.csv')

print(f"✓ Loaded data files:")
print(f"  - Recommendations: {len(recs_with_portions)} algorithms")
print(f"  - Test ratings: {len(test_df)} from {test_df['user_name'].nunique()} users")
print(f"  - Train ratings: {len(train_df)} from {train_df['user_name'].nunique()} users")

# CRITICAL FIX: Define the rated food universe
rated_foods = set(train_df['food'].unique()) | set(test_df['food'].unique())
print(f"\n✓ Rated food universe: {len(rated_foods)} unique foods")

# STEP 4.2: FILTER RECOMMENDATIONS TO RATED FOODS ONLY

print("\nSTEP 4.2: FILTERING RECOMMENDATIONS TO RATED FOODS")

filtered_recs = {}

for algorithm in recs_with_portions:
    filtered_recs[algorithm] = {}
    
    total_original = 0
    total_filtered = 0
    
    for user_name, recommendations in recs_with_portions[algorithm].items():
        # Keep only recommendations for foods that were rated somewhere
        filtered = [r for r in recommendations if r['food'] in rated_foods]
        
        filtered_recs[algorithm][user_name] = filtered
        total_original += len(recommendations)
        total_filtered += len(filtered)
    
    coverage = (total_filtered / total_original * 100) if total_original > 0 else 0
    print(f"  {algorithm:15s}: {total_filtered:4d}/{total_original:4d} recommendations kept ({coverage:.1f}%)")

print("\n⚠️  INTERPRETATION:")
print("  - 100% = Algorithm only recommends rated foods (good for evaluation)")
print("  - <100% = Algorithm recommends unrated foods (filtered out for fair comparison)")

# STEP 4.3: IMPLEMENT PREFERENCE METRICS

print("\nSTEP 4.3: IMPLEMENTING PREFERENCE METRICS")

K = 10
LIKE_THRESHOLD = 4

print(f"\nConfiguration:")
print(f"  K (top-K): {K}")
print(f"  Like threshold: rating ≥ {LIKE_THRESHOLD}")

def calculate_preference_metrics(recommendations, test_ratings, K=10, like_threshold=4):
    """
    Calculate preference metrics with proper handling of insufficient recommendations
    """
    # Get top K foods (or fewer if not enough available)
    rec_foods = [rec['food'] for rec in recommendations[:K]]
    num_available = len(rec_foods)
    
    # Get liked foods in test set
    liked_foods_in_test = {
        food for food, rating in test_ratings.items()
        if rating >= like_threshold
    }
    
    # Count hits
    hits = [food for food in rec_foods if food in liked_foods_in_test]
    
    # Precision: out of available recommendations, how many are liked?
    precision = len(hits) / num_available if num_available > 0 else 0
    
    # Recall: out of all liked foods, how many did we recommend?
    recall = len(hits) / len(liked_foods_in_test) if liked_foods_in_test else 0
    
    # F1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    # NDCG
    dcg = 0
    idcg = 0
    
    for i, food in enumerate(rec_foods):
        rating = test_ratings.get(food, 0)
        dcg += rating / np.log2(i + 2)
    
    ideal_ratings = sorted(test_ratings.values(), reverse=True)[:num_available]
    for i, rating in enumerate(ideal_ratings):
        idcg += rating / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # Hit rate
    hit_rate = 1 if len(hits) > 0 else 0
    
    # Coverage: did we get K recommendations?
    coverage = num_available / K
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ndcg': ndcg,
        'hit_rate': hit_rate,
        'num_hits': len(hits),
        'num_liked_in_test': len(liked_foods_in_test),
        'num_available': num_available,
        'coverage': coverage
    }

print("✓ Metrics implemented")

# STEP 4.4: EVALUATE ALL ALGORITHMS

print("\nSTEP 4.4: EVALUATING ALL ALGORITHMS")

preference_results = {}

for algorithm in filtered_recs:
    print(f"\n  Evaluating {algorithm}...")
    
    user_metrics = []
    users_with_insufficient_recs = 0
    
    for user_name, recommendations in filtered_recs[algorithm].items():
        # Get user's test ratings
        user_test = test_df[test_df['user_name'] == user_name]
        test_ratings = dict(zip(user_test['food'], user_test['rating']))
        
        if not test_ratings:
            continue
        
        # Calculate metrics
        metrics = calculate_preference_metrics(
            recommendations,
            test_ratings,
            K=K,
            like_threshold=LIKE_THRESHOLD
        )
        
        metrics['user_name'] = user_name
        user_metrics.append(metrics)
        
        if metrics['num_available'] < K:
            users_with_insufficient_recs += 1
    
    preference_results[algorithm] = pd.DataFrame(user_metrics)
    
    df = preference_results[algorithm]
    print(f"  ✓ Evaluated {len(user_metrics)} users")
    print(f"    Avg recommendations available: {df['num_available'].mean():.1f}/{K}")
    print(f"    Users with <{K} recommendations: {users_with_insufficient_recs}")
    print(f"    Avg F1@10: {df['f1'].mean():.4f}")

print("\n✓ All algorithms evaluated!")

# STEP 4.5: SUMMARY STATISTICS

print("\nSTEP 4.5: PREFERENCE EVALUATION SUMMARY")

summary_data = []

for algorithm in preference_results:
    df = preference_results[algorithm]
    
    summary = {
        'Algorithm': algorithm.replace('_', ' ').title(),
        'Precision@10': df['precision'].mean(),
        'Recall@10': df['recall'].mean(),
        'F1@10': df['f1'].mean(),
        'NDCG@10': df['ndcg'].mean(),
        'Hit Rate': df['hit_rate'].mean(),
        'Avg Coverage': df['coverage'].mean(),
        'Avg Hits': df['num_hits'].mean()
    }
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('F1@10', ascending=False)

print("\n" + "=" * 70)
print("PREFERENCE METRICS COMPARISON (RATED FOODS ONLY)")
print("=" * 70)
print("\n" + summary_df.to_string(index=False))

# Identify best algorithm
best_algorithm = summary_df.iloc[0]['Algorithm']
best_f1 = summary_df.iloc[0]['F1@10']

print(f"\n{'='*70}")
print(f"🏆 BEST ALGORITHM (by F1@10): {best_algorithm}")
print(f"   F1 Score: {best_f1:.4f}")
print(f"{'='*70}")

# STEP 4.6: SAVE RESULTS

print("\nSTEP 4.6: SAVING RESULTS")

with open('preference_evaluation_results.pkl', 'wb') as f:
    pickle.dump(preference_results, f)
print(f"✓ Saved: preference_evaluation_results.pkl")

summary_df.to_csv('preference_evaluation_summary.csv', index=False)
print(f"✓ Saved: preference_evaluation_summary.csv")

for algorithm in preference_results:
    filename = f'preference_results_{algorithm}.csv'
    preference_results[algorithm].to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")

# STEP 4.7: DETAILED ANALYSIS

print("\nSTEP 4.7: DETAILED ANALYSIS")

# Best algorithm breakdown
best_algo_key = best_algorithm.lower().replace(' ', '_')
if best_algo_key in preference_results:
    best_df = preference_results[best_algo_key]
    
    print(f"\nDetailed statistics for {best_algorithm}:")
    print(f"  Precision@10: {best_df['precision'].mean():.4f} ± {best_df['precision'].std():.4f}")
    print(f"  Recall@10:    {best_df['recall'].mean():.4f} ± {best_df['recall'].std():.4f}")
    print(f"  F1@10:        {best_df['f1'].mean():.4f} ± {best_df['f1'].std():.4f}")
    print(f"  NDCG@10:      {best_df['ndcg'].mean():.4f} ± {best_df['ndcg'].std():.4f}")
    print(f"  Hit Rate:     {best_df['hit_rate'].mean():.4f}")
    print(f"  Coverage:     {best_df['coverage'].mean():.4f}")
    
    print(f"\n  Average hits per user: {best_df['num_hits'].mean():.1f}")
    print(f"  Average liked foods in test: {best_df['num_liked_in_test'].mean():.1f}")

# Algorithm comparison table
print("\n" + "=" * 70)
print("ALGORITHM COMPARISON")
print("=" * 70)

comparison = []
for algo in preference_results:
    df = preference_results[algo]
    comparison.append({
        'Algorithm': algo.replace('_', ' ').title(),
        'Users': len(df),
        'Avg Recs': f"{df['num_available'].mean():.1f}/10",
        'Precision': f"{df['precision'].mean():.3f}",
        'Recall': f"{df['recall'].mean():.3f}",
        'F1': f"{df['f1'].mean():.3f}",
        'Hit%': f"{df['hit_rate'].mean()*100:.1f}%"
    })

comp_df = pd.DataFrame(comparison)
print("\n" + comp_df.to_string(index=False))

# FINAL SUMMARY

print("\n" + "=" * 70)
print("STAGE 4 COMPLETE ✓")
print("=" * 70)

print(f"""
📊 EVALUATION SUMMARY:

Methodology:
  • Filtered all algorithms to rated foods only (fair comparison)
  • Evaluated top-{K} recommendations per user
  • {len(test_df['user_name'].unique())} users evaluated
  • Like threshold: rating ≥ {LIKE_THRESHOLD}

Key Findings:
  • Best algorithm: {best_algorithm} (F1@10 = {best_f1:.4f})
  • This algorithm best understands user preferences for rated foods
  
Files Created:
  1. preference_evaluation_results.pkl
  2. preference_evaluation_summary.csv  
  3. preference_results_[algorithm].csv (per-algorithm details)

📝 INTERPRETATION FOR YOUR PROFESSOR:

Stage 4 answers: "Does the system understand the user's taste?"

Results show which algorithm best predicts user preferences when 
recommending from foods that users have previously rated.

⚠️  Note: Low absolute scores (F1 < 0.2) are common in food recommendation
   because user preferences are diverse and context-dependent.

➡️  NEXT: Stage 5 - Evaluate Nutritional Safety
   "Can the user safely eat the recommended portions?"
   This will complete the two-stage evaluation framework.
""")