import pandas as pd
import numpy as np
import pickle
import os

print("="*70)
print("STAGE 4: EVALUATE PREFERENCE ALIGNMENT (SELECTED + ALL)")
print("="*70)


# ============================================================
# STEP 4.1: LOAD DATA
# ============================================================

print("\nSTEP 4.1: LOADING DATA")

# Check required files
required_files = [
    'recommendations_all_algorithms_TFIDF.pkl',
    'data_test_ratings.csv'
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\n❌ ERROR: {file} not found!")
        print("Please run Stage 2 first.")
        exit()

# Load recommendations
with open('recommendations_all_algorithms_TFIDF.pkl', 'rb') as f:
    all_recommendations = pickle.load(f)

# Load test ratings
test_df = pd.read_csv('data_test_ratings.csv')

print(f"\nLoaded data:")
print(f"  - Algorithms: {len(all_recommendations)}")
print(f"  - Test ratings: {len(test_df)} ratings")


# ============================================================
# STEP 4.2: EVALUATION METRICS
# ============================================================

print("\n" + "="*70)
print("STEP 4.2: EVALUATION METRICS")
print("="*70)

LIKE_THRESHOLD = 3  # Rating >= 3 is considered "liked"

def precision_at_k(recommended_foods, liked_foods):
    """Precision@K: What fraction of recommendations were liked?"""
    if not recommended_foods:
        return 0.0
    
    hits = sum(1 for food in recommended_foods if food in liked_foods)
    return hits / len(recommended_foods)


def recall_at_k(recommended_foods, liked_foods):
    """Recall@K: What fraction of liked foods were recommended?"""
    if not liked_foods:
        return 0.0
    
    hits = sum(1 for food in recommended_foods if food in liked_foods)
    return hits / len(liked_foods)


def f1_score(precision, recall):
    """Harmonic mean of precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def hit_rate(recommended_foods, liked_foods):
    """Hit Rate: Did we recommend at least one liked food?"""
    if not recommended_foods:
        return 0.0
    
    return 1.0 if any(food in liked_foods for food in recommended_foods) else 0.0


print("✓ Evaluation metrics defined:")
print("  - Precision@K: Fraction of recommendations that were liked")
print("  - Recall@K: Fraction of liked foods that were recommended")
print("  - F1@K: Harmonic mean of precision and recall")
print("  - Hit Rate: Whether at least one recommendation was liked")


# ============================================================
# STEP 4.3: EVALUATE ALL ALGORITHMS
# ============================================================

print("\n" + "="*70)
print("STEP 4.3: EVALUATING ALL ALGORITHMS")
print("="*70)

results = {}

for algo_name, user_recommendations in all_recommendations.items():
    print(f"\nEvaluating {algo_name}...")
    
    algo_results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'hit_rate': []
    }
    
    for user_name, recs in user_recommendations.items():
        # Get test ratings for this user
        user_test = test_df[test_df['user_name'] == user_name]
        
        # Get liked foods (rating >= 3)
        liked_foods = set(user_test[user_test['rating'] >= LIKE_THRESHOLD]['food'].tolist())
        
        # Get recommended foods (just the food names, not scores)
        recommended_foods = [food for food, score in recs]
        
        # Calculate metrics
        prec = precision_at_k(recommended_foods, liked_foods)
        rec = recall_at_k(recommended_foods, liked_foods)
        f1 = f1_score(prec, rec)
        hr = hit_rate(recommended_foods, liked_foods)
        
        algo_results['precision'].append(prec)
        algo_results['recall'].append(rec)
        algo_results['f1'].append(f1)
        algo_results['hit_rate'].append(hr)
    
    # Calculate averages
    results[algo_name] = {
        'precision_avg': np.mean(algo_results['precision']) * 100,
        'recall_avg': np.mean(algo_results['recall']) * 100,
        'f1_avg': np.mean(algo_results['f1']) * 100,
        'hit_rate_avg': np.mean(algo_results['hit_rate']) * 100,
        'num_users': len(algo_results['precision'])
    }
    
    print(f"  F1@10: {results[algo_name]['f1_avg']:.1f}%")
    print(f"  Precision@10: {results[algo_name]['precision_avg']:.1f}%")
    print(f"  Recall@10: {results[algo_name]['recall_avg']:.1f}%")
    print(f"  Hit Rate: {results[algo_name]['hit_rate_avg']:.1f}%")


# ============================================================
# STEP 4.4: CREATE SUMMARY TABLE
# ============================================================

print("\n" + "="*70)
print("STEP 4.4: SUMMARY TABLE")
print("="*70)

# Create summary DataFrame
summary_data = []
for algo_name, metrics in results.items():
    summary_data.append({
        'Algorithm': algo_name,
        'F1@10 (%)': f"{metrics['f1_avg']:.1f}",
        'Precision@10 (%)': f"{metrics['precision_avg']:.1f}",
        'Recall@10 (%)': f"{metrics['recall_avg']:.1f}",
        'Hit Rate (%)': f"{metrics['hit_rate_avg']:.1f}",
        'Users': metrics['num_users']
    })

summary_df = pd.DataFrame(summary_data)

# Sort by F1 score
summary_df['F1_numeric'] = summary_df['F1@10 (%)'].astype(float)
summary_df = summary_df.sort_values('F1_numeric', ascending=False)
summary_df = summary_df.drop('F1_numeric', axis=1)

print("\n" + "="*70)
print("PREFERENCE EVALUATION RESULTS (NO SAFETY CONSTRAINTS)")
print("="*70)
print()
print(summary_df.to_string(index=False))
print()


# ============================================================
# STEP 4.5: COMPARISON: SELECTED VS ALL
# ============================================================

print("\n" + "="*70)
print("COMPARISON: SELECTED (SAME CUISINE) VS ALL (ANY CUISINE)")
print("="*70)

algorithms = ['content_based', 'collaborative', 'hybrid', 'popularity']

for algo in algorithms:
    selected_key = f"{algo}_selected"
    all_key = f"{algo}_all"
    
    if selected_key in results and all_key in results:
        selected_f1 = results[selected_key]['f1_avg']
        all_f1 = results[all_key]['f1_avg']
        
        print(f"\n{algo.upper()}:")
        print(f"  Selected (same cuisine): {selected_f1:.1f}% F1")
        print(f"  All (any cuisine): {all_f1:.1f}% F1")
        
        if selected_f1 > all_f1:
            improvement = selected_f1 - all_f1
            print(f"  → Selected is BETTER by {improvement:.1f} percentage points")
        elif all_f1 > selected_f1:
            improvement = all_f1 - selected_f1
            print(f"  → All is BETTER by {improvement:.1f} percentage points")
        else:
            print(f"  → TIE")


# ============================================================
# STEP 4.6: SAVE RESULTS
# ============================================================

print("\n" + "="*70)
print("STEP 4.6: SAVING RESULTS")
print("="*70)

# Save summary CSV
summary_df.to_csv('preference_evaluation_summary_TFIDF.csv', index=False)
print("✓ Saved: preference_evaluation_summary_TFIDF.csv")

# Save detailed results
with open('preference_evaluation_results_TFIDF.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✓ Saved: preference_evaluation_results_TFIDF.pkl")

print("\n" + "="*70)
print("STAGE 4 COMPLETE")
print("="*70)
print(f"""
KEY FINDINGS:
- Best Overall Algorithm: {summary_df.iloc[0]['Algorithm']}
  (F1@10: {summary_df.iloc[0]['F1@10 (%)']})

- Evaluated {len(all_recommendations)} algorithm variants
- Tested on {results[list(results.keys())[0]]['num_users']} users

Next: Run Stage 5 to evaluate preference + safety constraints
""")