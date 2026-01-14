import pandas as pd
import pickle

print("="*70)
print("ALL RESULTS SUMMARY")
print("="*70)

# Stage 4: Preference
print("\n📊 STAGE 4: PREFERENCE EVALUATION")
pref = pd.read_csv('preference_evaluation_summary_TFIDF.csv')
print(pref.to_string(index=False))

# Stage 5: Combined
print("\n📊 STAGE 5: LIKED & SAFE RATE")
liked_safe = pd.read_csv('stage5_perspective1_liked_and_safe_TFIDF.csv')
print(liked_safe.to_string(index=False))

# Stage 6: Stats
print("\n📊 STAGE 6: STATISTICAL SIGNIFICANCE")
print("\nCuisine Filtering Effect:")
stats = pd.read_csv('stage6_statistical_tests_preference.csv')
cuisine_tests = stats[stats['algorithm_2'].str.contains('_all')]
print(cuisine_tests[['algorithm_1', 'mean_difference', 'p_value', 'significant']].to_string(index=False))

print("\n✅ All files loaded successfully!")
print("\nFigures available:")
print("  - figure1_preference_metrics.png")
print("  - figure2_liked_and_safe.png") 
print("  - figure3_cuisine_filtering_effect.png")
print("  - figure4_three_perspectives.png")
print("  - figure5_summary_table.png")