import pandas as pd
import numpy as np
import pickle
import os

print("="*70)
print("STAGE 5: COMBINED PREFERENCE + SAFETY EVALUATION")
print("="*70)


# ============================================================
# STEP 5.1: LOAD DATA
# ============================================================

print("\nSTEP 5.1: LOADING DATA")

required_files = [
    'recommendations_with_portions_TFIDF.pkl',
    'data_test_ratings.csv'
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\n❌ ERROR: {file} not found!")
        print("Please run Stage 3 first.")
        exit()

# Load recommendations with safety data
with open('recommendations_with_portions_TFIDF.pkl', 'rb') as f:
    recommendations_with_safety = pickle.load(f)

# Load test ratings
test_df = pd.read_csv('data_test_ratings.csv')

print(f"\nLoaded data:")
print(f"  - Recommendations with safety: {len(recommendations_with_safety)} entries")
print(f"  - Test ratings: {len(test_df)} ratings")


# ============================================================
# STEP 5.2: ORGANIZE BY ALGORITHM AND USER
# ============================================================

print("\n" + "="*70)
print("STEP 5.2: ORGANIZING RECOMMENDATIONS BY ALGORITHM")
print("="*70)

# recommendations_with_safety is already organized as:
# {algorithm: {user: [list of recommendation dicts]}}

organized_recs = {}

for algo_name, users_dict in recommendations_with_safety.items():
    organized_recs[algo_name] = {}
    
    for user_name, recs_list in users_dict.items():
        # recs_list is already a list of dicts with keys:
        # 'food', 'score', 'portion_g', 'is_safe', etc.
        organized_recs[algo_name][user_name] = recs_list

print(f"✓ Organized {len(organized_recs)} algorithms")
for algo, users in organized_recs.items():
    total_recs = sum(len(recs) for recs in users.values())
    print(f"  {algo}: {total_recs} recommendations across {len(users)} users")


# ============================================================
# STEP 5.3: EVALUATION METRICS
# ============================================================

print("\n" + "="*70)
print("STEP 5.3: EVALUATION METRICS")
print("="*70)

LIKE_THRESHOLD = 3  # Rating >= 3 is considered "liked"


# ============================================================
# PERSPECTIVE 1: Liked & Safe Rate
# ============================================================

def evaluate_liked_and_safe_rate(algo_recs, test_df):
    """
    Of ALL recommendations made, what % are BOTH liked AND safe?
    """
    results = []
    
    for user_name, recs in algo_recs.items():
        # Get user's test ratings
        user_test = test_df[test_df['user_name'] == user_name]
        liked_foods = set(user_test[user_test['rating'] >= LIKE_THRESHOLD]['food'].tolist())
        
        for rec in recs:
            food = rec['food']
            is_safe = rec['is_safe']
            is_liked = food in liked_foods
            is_both = is_safe and is_liked
            
            results.append({
                'user': user_name,
                'food': food,
                'is_safe': is_safe,
                'is_liked': is_liked,
                'is_both': is_both
            })
    
    df = pd.DataFrame(results)
    
    total = len(df)
    safe_count = df['is_safe'].sum()
    liked_count = df['is_liked'].sum()
    both_count = df['is_both'].sum()
    
    return {
        'total_recommendations': total,
        'safe_recommendations': safe_count,
        'liked_recommendations': liked_count,
        'liked_and_safe_recommendations': both_count,
        'liked_and_safe_rate': (both_count / total * 100) if total > 0 else 0,
        'safe_rate': (safe_count / total * 100) if total > 0 else 0,
        'liked_rate': (liked_count / total * 100) if total > 0 else 0
    }


# ============================================================
# PERSPECTIVE 2: Coverage of Liked Foods
# ============================================================

def evaluate_coverage_of_liked_foods(algo_recs, test_df):
    """
    Of all liked foods in test set, what % did we find safely?
    """
    results = []
    
    for user_name, recs in algo_recs.items():
        # Get user's test ratings
        user_test = test_df[test_df['user_name'] == user_name]
        liked_foods = set(user_test[user_test['rating'] >= LIKE_THRESHOLD]['food'].tolist())
        
        # Get safe recommendations
        safe_recs = [rec['food'] for rec in recs if rec['is_safe']]
        
        # Find liked foods that were recommended safely
        found_safely = set(safe_recs) & liked_foods
        
        results.append({
            'user': user_name,
            'total_liked_foods': len(liked_foods),
            'found_safely': len(found_safely),
            'coverage_rate': (len(found_safely) / len(liked_foods) * 100) if liked_foods else 0
        })
    
    df = pd.DataFrame(results)
    
    total_liked = df['total_liked_foods'].sum()
    total_found = df['found_safely'].sum()
    
    return {
        'total_liked_foods_in_test': total_liked,
        'found_safely': total_found,
        'coverage_rate': (total_found / total_liked * 100) if total_liked > 0 else 0,
        'avg_coverage_per_user': df['coverage_rate'].mean()
    }


# ============================================================
# PERSPECTIVE 3: Safety-First Acceptance Rate
# ============================================================

def evaluate_safety_first_acceptance(algo_recs, test_df):
    """
    Of recommendations that ARE safe, what % are liked?
    (Quality of safe recommendations)
    """
    results = []
    
    for user_name, recs in algo_recs.items():
        # Get user's test ratings
        user_test = test_df[test_df['user_name'] == user_name]
        liked_foods = set(user_test[user_test['rating'] >= LIKE_THRESHOLD]['food'].tolist())
        
        # Get only safe recommendations
        safe_recs = [rec for rec in recs if rec['is_safe']]
        
        for rec in safe_recs:
            food = rec['food']
            is_liked = food in liked_foods
            
            results.append({
                'user': user_name,
                'food': food,
                'is_liked': is_liked
            })
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        return {
            'total_safe_recommendations': 0,
            'liked_among_safe': 0,
            'acceptance_rate': 0
        }
    
    total_safe = len(df)
    liked_count = df['is_liked'].sum()
    
    return {
        'total_safe_recommendations': total_safe,
        'liked_among_safe': liked_count,
        'acceptance_rate': (liked_count / total_safe * 100) if total_safe > 0 else 0
    }


print("✓ Three evaluation perspectives defined:")
print("  1. Liked & Safe Rate: % of all recommendations that are both")
print("  2. Coverage: % of liked foods we found safely")
print("  3. Safety-First Acceptance: % of safe recs that are liked")


# ============================================================
# STEP 5.4: EVALUATE ALL ALGORITHMS
# ============================================================

print("\n" + "="*70)
print("STEP 5.4: EVALUATING ALL ALGORITHMS")
print("="*70)

all_results = {}

for algo_name, user_recs in organized_recs.items():
    print(f"\nEvaluating {algo_name}...")
    
    # Perspective 1
    p1 = evaluate_liked_and_safe_rate(user_recs, test_df)
    
    # Perspective 2
    p2 = evaluate_coverage_of_liked_foods(user_recs, test_df)
    
    # Perspective 3
    p3 = evaluate_safety_first_acceptance(user_recs, test_df)
    
    all_results[algo_name] = {
        'perspective_1': p1,
        'perspective_2': p2,
        'perspective_3': p3
    }
    
    print(f"  Liked & Safe: {p1['liked_and_safe_recommendations']}/{p1['total_recommendations']} ({p1['liked_and_safe_rate']:.1f}%)")
    print(f"  Coverage: {p2['found_safely']}/{p2['total_liked_foods_in_test']} ({p2['coverage_rate']:.1f}%)")
    print(f"  Acceptance: {p3['liked_among_safe']}/{p3['total_safe_recommendations']} ({p3['acceptance_rate']:.1f}%)")


# ============================================================
# STEP 5.5: CREATE SUMMARY TABLES
# ============================================================

print("\n" + "="*70)
print("STEP 5.5: SUMMARY TABLES")
print("="*70)

# Table 1: Liked & Safe Rate
print("\n" + "="*70)
print("PERSPECTIVE 1: LIKED & SAFE RATE")
print("="*70)

summary_p1 = []
for algo, results in all_results.items():
    p1 = results['perspective_1']
    summary_p1.append({
        'Algorithm': algo,
        'Total Recs': p1['total_recommendations'],
        'Liked & Safe': p1['liked_and_safe_recommendations'],
        'Rate (%)': f"{p1['liked_and_safe_rate']:.1f}"
    })

df_p1 = pd.DataFrame(summary_p1)
df_p1['Rate_numeric'] = df_p1['Rate (%)'].astype(float)
df_p1 = df_p1.sort_values('Rate_numeric', ascending=False)
df_p1 = df_p1.drop('Rate_numeric', axis=1)
print()
print(df_p1.to_string(index=False))
print()


# Table 2: Coverage
print("\n" + "="*70)
print("PERSPECTIVE 2: COVERAGE OF LIKED FOODS")
print("="*70)

summary_p2 = []
for algo, results in all_results.items():
    p2 = results['perspective_2']
    summary_p2.append({
        'Algorithm': algo,
        'Liked Foods in Test': p2['total_liked_foods_in_test'],
        'Found Safely': p2['found_safely'],
        'Coverage (%)': f"{p2['coverage_rate']:.1f}"
    })

df_p2 = pd.DataFrame(summary_p2)
df_p2['Coverage_numeric'] = df_p2['Coverage (%)'].astype(float)
df_p2 = df_p2.sort_values('Coverage_numeric', ascending=False)
df_p2 = df_p2.drop('Coverage_numeric', axis=1)
print()
print(df_p2.to_string(index=False))
print()


# Table 3: Safety-First Acceptance
print("\n" + "="*70)
print("PERSPECTIVE 3: SAFETY-FIRST ACCEPTANCE RATE")
print("="*70)

summary_p3 = []
for algo, results in all_results.items():
    p3 = results['perspective_3']
    summary_p3.append({
        'Algorithm': algo,
        'Safe Recs': p3['total_safe_recommendations'],
        'Liked': p3['liked_among_safe'],
        'Acceptance (%)': f"{p3['acceptance_rate']:.1f}"
    })

df_p3 = pd.DataFrame(summary_p3)
df_p3['Acceptance_numeric'] = df_p3['Acceptance (%)'].astype(float)
df_p3 = df_p3.sort_values('Acceptance_numeric', ascending=False)
df_p3 = df_p3.drop('Acceptance_numeric', axis=1)
print()
print(df_p3.to_string(index=False))
print()


# ============================================================
# STEP 5.6: SELECTED VS ALL COMPARISON
# ============================================================

print("\n" + "="*70)
print("COMPARISON: SELECTED (SAME CUISINE) VS ALL (ANY CUISINE)")
print("="*70)

algorithms = ['content_based', 'collaborative', 'hybrid', 'popularity']

for algo in algorithms:
    selected_key = f"{algo}_selected"
    all_key = f"{algo}_all"
    
    if selected_key in all_results and all_key in all_results:
        selected_rate = all_results[selected_key]['perspective_1']['liked_and_safe_rate']
        all_rate = all_results[all_key]['perspective_1']['liked_and_safe_rate']
        
        print(f"\n{algo.upper()} - Liked & Safe Rate:")
        print(f"  Selected (same cuisine): {selected_rate:.1f}%")
        print(f"  All (any cuisine): {all_rate:.1f}%")
        
        if selected_rate > all_rate:
            improvement = selected_rate - all_rate
            print(f"  → Selected is BETTER by {improvement:.1f} percentage points")
        elif all_rate > selected_rate:
            improvement = all_rate - selected_rate
            print(f"  → All is BETTER by {improvement:.1f} percentage points")
        else:
            print(f"  → TIE")


# ============================================================
# STEP 5.7: SAVE RESULTS
# ============================================================

print("\n" + "="*70)
print("STEP 5.7: SAVING RESULTS")
print("="*70)

# Save all three perspectives
df_p1.to_csv('stage5_perspective1_liked_and_safe_TFIDF.csv', index=False)
print("✓ Saved: stage5_perspective1_liked_and_safe_TFIDF.csv")

df_p2.to_csv('stage5_perspective2_coverage_TFIDF.csv', index=False)
print("✓ Saved: stage5_perspective2_coverage_TFIDF.csv")

df_p3.to_csv('stage5_perspective3_acceptance_TFIDF.csv', index=False)
print("✓ Saved: stage5_perspective3_acceptance_TFIDF.csv")

# Save detailed results
with open('stage5_detailed_results_TFIDF.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("✓ Saved: stage5_detailed_results_TFIDF.pkl")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("STAGE 5 COMPLETE - FINAL RESULTS")
print("="*70)

# Find best in each perspective
best_p1 = df_p1.iloc[0]
best_p2 = df_p2.iloc[0]
best_p3 = df_p3.iloc[0]

print(f"""
BEST ALGORITHMS BY PERSPECTIVE:

1. LIKED & SAFE RATE (Primary Metric):
   Winner: {best_p1['Algorithm']}
   Rate: {best_p1['Rate (%)']}% ({best_p1['Liked & Safe']}/{best_p1['Total Recs']} recommendations)

2. COVERAGE OF LIKED FOODS:
   Winner: {best_p2['Algorithm']}
   Coverage: {best_p2['Coverage (%)']}% ({best_p2['Found Safely']}/{best_p2['Liked Foods in Test']} foods)

3. SAFETY-FIRST ACCEPTANCE:
   Winner: {best_p3['Algorithm']}
   Acceptance: {best_p3['Acceptance (%)']}% ({best_p3['Liked']}/{best_p3['Safe Recs']} safe recs liked)

EVALUATION COMPLETE
All results saved to CSV files for analysis.
""")