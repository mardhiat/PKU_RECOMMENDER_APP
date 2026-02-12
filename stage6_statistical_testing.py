import pandas as pd
import numpy as np
from scipy import stats
import pickle

 
print("STAGE 6: STATISTICAL SIGNIFICANCE TESTING")
 

 # LOAD RESULTS FROM STAGE 4 AND 5
 
print("\nLOADING EVALUATION RESULTS...")

# Load Stage 4 results (preference)
with open('preference_evaluation_results_TFIDF.pkl', 'rb') as f:
    stage4_results = pickle.load(f)

# Load Stage 5 results (preference + safety)
with open('stage5_detailed_results_TFIDF.pkl', 'rb') as f:
    stage5_results = pickle.load(f)

print(f"✓ Loaded Stage 4 results (preference evaluation)")
print(f"✓ Loaded Stage 5 results (combined evaluation)")

 # PREPARE PER-USER METRICS
 
 
print("PREPARING PER-USER METRICS")
 

algorithms = [
    'content_based_selected',
    'content_based_all',
    'collaborative_selected',
    'collaborative_all',
    'hybrid_selected',
    'hybrid_all',
    'popularity_selected',
    'popularity_all',
    'random'
]

# Extract per-user F1 scores from Stage 4
print("\nExtracting per-user F1 scores (preference)...")
stage4_per_user = {}
for algo in algorithms:
    if algo in stage4_results and 'per_user_f1' in stage4_results[algo]:
        stage4_per_user[algo] = stage4_results[algo]['per_user_f1']

print(f"✓ Extracted F1 scores for {len(stage4_per_user)} algorithms")

# Extract per-user liked+safe rates from Stage 5
print("\nExtracting per-user liked+safe rates (preference + safety)...")
stage5_per_user = {}

for algo in algorithms:
    if algo in stage5_results and 'per_user_liked_safe' in stage5_results[algo]:
        stage5_per_user[algo] = stage5_results[algo]['per_user_liked_safe']

print(f"✓ Extracted liked+safe rates for {len(stage5_per_user)} algorithms")

 # STATISTICAL TESTING FUNCTIONS
 
def paired_t_test(scores1, scores2, name1, name2):
    """
    Perform paired t-test between two algorithms
    Returns: t-statistic, p-value, mean difference
    """
    # Ensure same length
    min_len = min(len(scores1), len(scores2))
    scores1 = scores1[:min_len]
    scores2 = scores2[:min_len]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Mean difference
    mean_diff = np.mean(scores1) - np.mean(scores2)
    
    return {
        'algorithm_1': name1,
        'algorithm_2': name2,
        'mean_1': np.mean(scores1),
        'mean_2': np.mean(scores2),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def cohens_d(scores1, scores2):
    """
    Calculate Cohen's d effect size
    Small: 0.2, Medium: 0.5, Large: 0.8
    """
    n1, n2 = len(scores1), len(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    # Interpret effect size
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return d, interpretation


 # STAGE 4: PREFERENCE EVALUATION SIGNIFICANCE
 
 
print("STAGE 4: PREFERENCE EVALUATION - STATISTICAL TESTS")
 

# Key comparisons for Stage 4
comparisons_stage4 = [
    ('hybrid_selected', 'collaborative_selected'),  # Best two
    ('hybrid_selected', 'content_based_selected'),  # Hybrid vs content
    ('collaborative_selected', 'content_based_selected'),  # Collab vs content
    ('content_based_selected', 'content_based_all'),  # Cuisine filtering effect
    ('collaborative_selected', 'collaborative_all'),  # Cuisine filtering effect
    ('hybrid_selected', 'hybrid_all'),  # Cuisine filtering effect
    ('hybrid_selected', 'popularity_selected'),  # Best vs baseline
    ('hybrid_selected', 'random'),  # Best vs random
]

stage4_test_results = []

print("\nRunning paired t-tests on F1 scores...")
print("-" * 70)

for algo1, algo2 in comparisons_stage4:
    if algo1 in stage4_per_user and algo2 in stage4_per_user:
        # T-test
        result = paired_t_test(
            stage4_per_user[algo1],
            stage4_per_user[algo2],
            algo1,
            algo2
        )
        
        # Cohen's d
        d, interpretation = cohens_d(
            stage4_per_user[algo1],
            stage4_per_user[algo2]
        )
        
        result['cohens_d'] = d
        result['effect_size'] = interpretation
        
        stage4_test_results.append(result)
        
        # Print result
        sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
        print(f"\n{algo1} vs {algo2}:")
        print(f"  Mean difference: {result['mean_difference']:+.2f}%")
        print(f"  p-value: {result['p_value']:.4f} {sig_marker}")
        print(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']})")
        if result['significant']:
            winner = algo1 if result['mean_difference'] > 0 else algo2
            print(f"  → {winner} is SIGNIFICANTLY better")
        else:
            print(f"  → No significant difference")

# Save Stage 4 results
stage4_df = pd.DataFrame(stage4_test_results)
stage4_df.to_csv('stage6_statistical_tests_preference.csv', index=False)
print(f"\n✓ Saved: stage6_statistical_tests_preference.csv")


 # STAGE 5: COMBINED EVALUATION SIGNIFICANCE
 
 
print("STAGE 5: COMBINED EVALUATION - STATISTICAL TESTS")
 

# Key comparisons for Stage 5 (same as Stage 4)
comparisons_stage5 = comparisons_stage4

stage5_test_results = []

print("\nRunning paired t-tests on liked+safe rates...")
print("-" * 70)

for algo1, algo2 in comparisons_stage5:
    if algo1 in stage5_per_user and algo2 in stage5_per_user:
        # T-test
        result = paired_t_test(
            stage5_per_user[algo1],
            stage5_per_user[algo2],
            algo1,
            algo2
        )
        
        # Cohen's d
        d, interpretation = cohens_d(
            stage5_per_user[algo1],
            stage5_per_user[algo2]
        )
        
        result['cohens_d'] = d
        result['effect_size'] = interpretation
        
        stage5_test_results.append(result)
        
        # Print result
        sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
        print(f"\n{algo1} vs {algo2}:")
        print(f"  Mean difference: {result['mean_difference']:+.2f}%")
        print(f"  p-value: {result['p_value']:.4f} {sig_marker}")
        print(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']})")
        if result['significant']:
            winner = algo1 if result['mean_difference'] > 0 else algo2
            print(f"  → {winner} is SIGNIFICANTLY better")
        else:
            print(f"  → No significant difference")

# Save Stage 5 results
stage5_df = pd.DataFrame(stage5_test_results)
stage5_df.to_csv('stage6_statistical_tests_combined.csv', index=False)
print(f"\n✓ Saved: stage6_statistical_tests_combined.csv")


 # SUMMARY OF KEY FINDINGS
 
 
print("KEY STATISTICAL FINDINGS")
 

print("\n1. BEST ALGORITHM COMPARISONS (PREFERENCE):")
print("-" * 70)

# Hybrid vs Collaborative
hvc_pref = next((r for r in stage4_test_results if 
                 r['algorithm_1'] == 'hybrid_selected' and 
                 r['algorithm_2'] == 'collaborative_selected'), None)
if hvc_pref:
    if hvc_pref['significant']:
        winner = 'Hybrid' if hvc_pref['mean_difference'] > 0 else 'Collaborative'
        print(f"✓ {winner} is SIGNIFICANTLY better (p={hvc_pref['p_value']:.4f})")
        print(f"  Effect size: {hvc_pref['effect_size']} (d={hvc_pref['cohens_d']:.3f})")
    else:
        print(f"✗ No significant difference between Hybrid and Collaborative")
        print(f"  (p={hvc_pref['p_value']:.4f}, difference too small)")

print("\n2. BEST ALGORITHM COMPARISONS (COMBINED):")
print("-" * 70)

# Hybrid vs Collaborative
hvc_comb = next((r for r in stage5_test_results if 
                 r['algorithm_1'] == 'hybrid_selected' and 
                 r['algorithm_2'] == 'collaborative_selected'), None)
if hvc_comb:
    if hvc_comb['significant']:
        winner = 'Hybrid' if hvc_comb['mean_difference'] > 0 else 'Collaborative'
        print(f"✓ {winner} is SIGNIFICANTLY better (p={hvc_comb['p_value']:.4f})")
        print(f"  Effect size: {hvc_comb['effect_size']} (d={hvc_comb['cohens_d']:.3f})")
    else:
        print(f"✗ No significant difference between Hybrid and Collaborative")
        print(f"  (p={hvc_comb['p_value']:.4f}, difference too small)")

print("\n3. CUISINE FILTERING EFFECT:")
print("-" * 70)

# Check if Selected > All for each algorithm
cuisine_effects = []
for base_algo in ['content_based', 'collaborative', 'hybrid']:
    selected = f"{base_algo}_selected"
    all_algo = f"{base_algo}_all"
    
    # Preference
    pref_result = next((r for r in stage4_test_results if 
                       r['algorithm_1'] == selected and 
                       r['algorithm_2'] == all_algo), None)
    
    # Combined
    comb_result = next((r for r in stage5_test_results if 
                       r['algorithm_1'] == selected and 
                       r['algorithm_2'] == all_algo), None)
    
    if pref_result and comb_result:
        print(f"\n{base_algo.upper()}:")
        print(f"  Preference: {'+' if pref_result['mean_difference'] > 0 else ''}{pref_result['mean_difference']:.2f}% "
              f"(p={pref_result['p_value']:.4f}) {'***' if pref_result['significant'] else 'ns'}")
        print(f"  Combined:   {'+' if comb_result['mean_difference'] > 0 else ''}{comb_result['mean_difference']:.2f}% "
              f"(p={comb_result['p_value']:.4f}) {'***' if comb_result['significant'] else 'ns'}")

print("\n4. BEATING BASELINES:")
print("-" * 70)

# Hybrid vs baselines
for baseline in ['popularity_selected', 'random']:
    pref_result = next((r for r in stage4_test_results if 
                       r['algorithm_1'] == 'hybrid_selected' and 
                       r['algorithm_2'] == baseline), None)
    
    if pref_result:
        print(f"\nHybrid vs {baseline}:")
        print(f"  Difference: +{pref_result['mean_difference']:.2f}%")
        print(f"  p-value: {pref_result['p_value']:.4f}")
        print(f"  Effect size: {pref_result['effect_size']} (d={pref_result['cohens_d']:.3f})")
        if pref_result['significant']:
            print(f"  ✓ SIGNIFICANTLY better than baseline")


 # WRITE SUMMARY FOR THESIS
 
 
print("THESIS-READY SUMMARY")
 

summary = []

# Find best algorithm
best_pref = max([(algo, np.mean(scores)) for algo, scores in stage4_per_user.items()], 
                key=lambda x: x[1])
best_comb = max([(algo, np.mean(scores)) for algo, scores in stage5_per_user.items()], 
                key=lambda x: x[1])

summary.append(f"\nBEST ALGORITHM:")
summary.append(f"  Preference: {best_pref[0]} (F1={best_pref[1]:.1f}%)")
summary.append(f"  Combined: {best_comb[0]} (Liked+Safe={best_comb[1]:.1f}%)")

# Cuisine filtering
summary.append(f"\nCUISINE FILTERING CONTRIBUTION:")
for base_algo in ['content_based', 'collaborative', 'hybrid']:
    selected = f"{base_algo}_selected"
    all_algo = f"{base_algo}_all"
    
    if selected in stage4_per_user and all_algo in stage4_per_user:
        diff = np.mean(stage4_per_user[selected]) - np.mean(stage4_per_user[all_algo])
        result = next((r for r in stage4_test_results if 
                      r['algorithm_1'] == selected and 
                      r['algorithm_2'] == all_algo), None)
        if result:
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
            summary.append(f"  {base_algo}: +{diff:.1f}% (p={result['p_value']:.4f}){sig}")

# Statistical significance
summary.append(f"\nSTATISTICAL SIGNIFICANCE:")
sig_count = sum(1 for r in stage4_test_results if r['significant'])
summary.append(f"  {sig_count}/{len(stage4_test_results)} comparisons are significant (p<0.05)")

print("\n".join(summary))

# Save summary
with open('stage6_thesis_summary.txt', 'w') as f:
    f.write("\n".join(summary))
print(f"\n✓ Saved: stage6_thesis_summary.txt")


 
print("STAGE 6 COMPLETE")
 
print(f"""
Statistical testing complete!

FILES CREATED:
  ✓ stage6_statistical_tests_preference.csv
  ✓ stage6_statistical_tests_combined.csv
  ✓ stage6_thesis_summary.txt

KEY FINDINGS:
  • Tested {len(stage4_test_results)} algorithm comparisons
  • Evaluated both preference and combined metrics
  • Calculated p-values and effect sizes (Cohen's d)
  • Identified statistically significant differences

SIGNIFICANCE LEVELS:
  *** p < 0.001 (highly significant)
  **  p < 0.01  (very significant)
  *   p < 0.05  (significant)
  ns  p ≥ 0.05  (not significant)

NEXT: Use these results in your thesis to justify algorithm selection
""")