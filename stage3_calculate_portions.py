import pandas as pd
import numpy as np
import pickle
import os

print("STAGE 3: CALCULATE SAFE PORTIONS")

# ============================================================
# STEP 3.1: LOAD DATA FROM PREVIOUS STAGES
# ============================================================

print("STEP 3.1: LOADING DATA")

# Check if required files exist
required_files = [
    'recommendations_all_algorithms.pkl',
    'data_food_database.csv',
    'data_user_nutritional_limits.csv'
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\n❌ ERROR: {file} not found!")
        print("Please run previous stages first.")
        exit()

# Load recommendations
with open('recommendations_all_algorithms.pkl', 'rb') as f:
    recommendations_dict = pickle.load(f)

# Load food database
food_db_df = pd.read_csv('data_food_database.csv')

# Load user limits
user_limits_df = pd.read_csv('data_user_nutritional_limits.csv')

print(f"\nLoaded data files:")
print(f"  - Recommendations: {len(recommendations_dict)} algorithms")
print(f"  - Food database: {len(food_db_df)} foods")
print(f"  - User limits: {len(user_limits_df)} users")

# Create food database dictionary
food_db = {}
for _, row in food_db_df.iterrows():
    food_db[row['food_name']] = {
        'phe_mg_per_100g': row['phe_mg_per_100g'],
        'protein_g_per_100g': row['protein_g_per_100g'],
        'energy_kcal_per_100g': row['energy_kcal_per_100g'],
        'serving_size_g': row.get('serving_size_g', 100.0)
    }

# Create user limits dictionary
user_limits = {}
for _, row in user_limits_df.iterrows():
    user_limits[row['user_name']] = {
        'phe_mg_min': row['phe_mg_min'],
        'phe_mg_max': row['phe_mg_max'],
        'protein_g': row['protein_g'],
        'energy_kcal': row['energy_kcal']
    }

print(f"Dictionaries created for fast lookup")


# ============================================================
# STEP 3.2: IMPLEMENT IMPROVED PORTION CALCULATION
# ============================================================

print("STEP 3.2: IMPLEMENTING PORTION CALCULATION")

# Configuration parameters
MEAL_FRACTION = 0.33  # Each meal is 1/3 of daily limit
PHE_MIN_RELAXATION = 0.5  # FIXED: Only require 50% of PHE minimum for single meal
MIN_PRACTICAL_PORTION = 30  # Minimum grams for a practical serving
MAX_PORTION_CAP = 500  # Maximum grams for practicality
ENERGY_TOLERANCE = 0.5  # Allow ±50% of energy target

print(f"\nConfiguration:")
print(f"  Meal fraction: {MEAL_FRACTION*100:.0f}% of daily limit")
print(f"  PHE minimum relaxation: {PHE_MIN_RELAXATION*100:.0f}% (FIXED)")
print(f"  Minimum practical portion: {MIN_PRACTICAL_PORTION}g")
print(f"  Maximum portion cap: {MAX_PORTION_CAP}g")


def calculate_safe_portion(food_nutrients, user_daily_limits, 
                          meal_fraction=0.33, phe_min_relaxation=0.5,
                          min_practical=30, max_cap=500, energy_tolerance=0.5):
    """
    Calculate maximum safe portion for a food given user's limits
    IMPROVED VERSION: Considers PHE min/max range, protein ceiling, energy target
    FIXED: Relaxed PHE minimum requirement for individual meals
    
    Args:
        food_nutrients: dict with nutrients per 100g
        user_daily_limits: dict with user's daily limits
        meal_fraction: fraction of daily limit for this meal
        phe_min_relaxation: multiplier for PHE minimum (0.5 = require only 50%)
        min_practical: minimum practical portion size (grams)
        max_cap: maximum portion size (grams)
        energy_tolerance: acceptable deviation from energy target (fraction)
    
    Returns:
        dict with portion_g, nutrients, is_practical, is_safe, limiting_factor
    """
    # Calculate meal targets
    # FIXED: Relax PHE minimum since users get PHE from other meals
    meal_phe_min = user_daily_limits['phe_mg_min'] * meal_fraction * phe_min_relaxation
    meal_phe_max = user_daily_limits['phe_mg_max'] * meal_fraction
    meal_protein_max = user_daily_limits['protein_g'] * meal_fraction
    meal_energy_target = user_daily_limits['energy_kcal'] * meal_fraction
    meal_energy_min = meal_energy_target * (1 - energy_tolerance)
    meal_energy_max = meal_energy_target * (1 + energy_tolerance)
    
    # Calculate nutrients per gram
    phe_per_g = food_nutrients['phe_mg_per_100g'] / 100.0
    protein_per_g = food_nutrients['protein_g_per_100g'] / 100.0
    energy_per_g = food_nutrients['energy_kcal_per_100g'] / 100.0
    
    # === CALCULATE PORTION BOUNDS ===
    
    # PHE constraints
    if phe_per_g > 0:
        min_portion_phe = meal_phe_min / phe_per_g  # Need at least this for PHE
        max_portion_phe = meal_phe_max / phe_per_g  # Can't exceed this for PHE
    else:
        min_portion_phe = 0
        max_portion_phe = float('inf')
    
    # Protein constraint (ceiling only)
    if protein_per_g > 0:
        max_portion_protein = meal_protein_max / protein_per_g
    else:
        max_portion_protein = float('inf')
    
    # Energy target (for optimization)
    if energy_per_g > 0:
        ideal_portion_energy = meal_energy_target / energy_per_g
    else:
        ideal_portion_energy = max_cap
    
    # === DETERMINE SAFE PORTION RANGE ===
    
    # Minimum safe portion: meet PHE minimum, be practical
    safe_portion_min = max(min_portion_phe, min_practical)
    
    # Maximum safe portion: don't exceed PHE max, protein max, or cap
    safe_portion_max = min(max_portion_phe, max_portion_protein, max_cap)
    
    # === CHECK FEASIBILITY ===
    
    if safe_portion_min > safe_portion_max:
        # Cannot meet PHE minimum without exceeding protein/PHE maximum
        return {
            'safe_portion_g': 0,
            'phe_mg': 0,
            'protein_g': 0,
            'energy_kcal': 0,
            'is_practical': False,
            'is_safe': False,
            'limiting_factor': 'infeasible',
            'failure_reason': 'Cannot meet PHE minimum without exceeding limits'
        }
    
    # === OPTIMIZE PORTION FOR ENERGY TARGET ===
    
    # Try to get close to energy target while staying in safe range
    recommended_portion = np.clip(ideal_portion_energy, safe_portion_min, safe_portion_max)
    
    # === CALCULATE ACTUAL NUTRIENTS ===
    
    actual_phe = phe_per_g * recommended_portion
    actual_protein = protein_per_g * recommended_portion
    actual_energy = energy_per_g * recommended_portion
    
    # === SAFETY CHECKS ===
    
    phe_safe = meal_phe_min <= actual_phe <= meal_phe_max
    protein_safe = actual_protein <= meal_protein_max
    energy_acceptable = meal_energy_min <= actual_energy <= meal_energy_max
    is_practical = recommended_portion >= min_practical
    
    # Overall safety: meets all constraints
    is_safe = phe_safe and protein_safe and is_practical
    
    # Determine limiting factor
    if max_portion_phe < max_portion_protein:
        limiting_factor = 'PHE'
    elif max_portion_protein < max_portion_phe:
        limiting_factor = 'PROTEIN'
    else:
        limiting_factor = 'BOTH'
    
    # === RETURN RESULTS ===
    
    return {
        'safe_portion_g': recommended_portion,
        'phe_mg': actual_phe,
        'protein_g': actual_protein,
        'energy_kcal': actual_energy,
        'is_practical': is_practical,
        'is_safe': is_safe,
        'phe_safe': phe_safe,
        'protein_safe': protein_safe,
        'energy_acceptable': energy_acceptable,
        'limiting_factor': limiting_factor,
        'phe_utilization': actual_phe / meal_phe_max if meal_phe_max > 0 else 0,
        'protein_utilization': actual_protein / meal_protein_max if meal_protein_max > 0 else 0,
        'energy_match': abs(actual_energy - meal_energy_target) / meal_energy_target if meal_energy_target > 0 else 0
    }

print("Portion calculation function implemented")


# ============================================================
# STEP 3.3: CALCULATE PORTIONS FOR ALL RECOMMENDATIONS
# ============================================================

print("\nSTEP 3.3: CALCULATING PORTIONS FOR ALL RECOMMENDATIONS")

# Count total recommendations to process
total_recs = sum(
    len(recs) for algorithm in recommendations_dict.values()
    for recs in algorithm.values()
)

print(f"\nTotal recommendations to process: {total_recs}")
print("Processing...")

# Store recommendations with portions
recommendations_with_portions = {}

processed = 0
skipped = 0

for algorithm in recommendations_dict:
    print(f"\n  Processing {algorithm}...")
    recommendations_with_portions[algorithm] = {}
    
    for user_name, recommendations in recommendations_dict[algorithm].items():
        # Get user's limits
        if user_name not in user_limits:
            print(f"    ⚠ No limits for user {user_name}, skipping...")
            skipped += len(recommendations)
            continue
        
        user_daily_limits = user_limits[user_name]
        
        # Process each recommendation
        recs_with_portions = []
        
        for food_name, score in recommendations:
            # Clean food name (remove cuisine tag)
            clean_name = food_name.lower().strip()
            if '(' in clean_name:
                clean_name = clean_name.split('(')[0].strip()
            
            # Get food nutrients
            food_nutrients = food_db.get(clean_name)
            
            if food_nutrients is None:
                # Skip foods without nutritional data
                skipped += 1
                continue
            
            # Calculate safe portion
            portion_info = calculate_safe_portion(
                food_nutrients,
                user_daily_limits,
                meal_fraction=MEAL_FRACTION,
                phe_min_relaxation=PHE_MIN_RELAXATION,
                min_practical=MIN_PRACTICAL_PORTION,
                max_cap=MAX_PORTION_CAP,
                energy_tolerance=ENERGY_TOLERANCE
            )
            
            # Store recommendation with portion info
            recs_with_portions.append({
                'food': food_name,
                'score': score,
                'portion_g': portion_info['safe_portion_g'],
                'phe_mg': portion_info['phe_mg'],
                'protein_g': portion_info['protein_g'],
                'energy_kcal': portion_info['energy_kcal'],
                'is_practical': portion_info['is_practical'],
                'is_safe': portion_info['is_safe'],
                'limiting_factor': portion_info['limiting_factor'],
                'phe_utilization': portion_info.get('phe_utilization', 0),
                'protein_utilization': portion_info.get('protein_utilization', 0)
            })
            
            processed += 1
        
        recommendations_with_portions[algorithm][user_name] = recs_with_portions

print(f"\nProcessing complete!")
print(f"  Processed: {processed} recommendations")
print(f"  Skipped (no nutrition data): {skipped} recommendations")


# ============================================================
# STEP 3.4: ANALYZE PORTION STATISTICS
# ============================================================

print("\nSTEP 3.4: ANALYZING PORTION STATISTICS")

# Collect statistics
all_portions = []
all_practical = []
all_safe = []
limiting_factors = {'PHE': 0, 'PROTEIN': 0, 'BOTH': 0, 'infeasible': 0}

for algorithm in recommendations_with_portions:
    for user_name, recs in recommendations_with_portions[algorithm].items():
        for rec in recs:
            all_portions.append(rec['portion_g'])
            all_practical.append(rec['is_practical'])
            all_safe.append(rec['is_safe'])
            limiting_factors[rec['limiting_factor']] += 1

if all_portions:
    print(f"\nPortion size statistics (across all recommendations):")
    print(f"  Mean: {np.mean(all_portions):.1f}g")
    print(f"  Median: {np.median(all_portions):.1f}g")
    print(f"  Min: {np.min(all_portions):.1f}g")
    print(f"  Max: {np.max(all_portions):.1f}g")
    print(f"  Std: {np.std(all_portions):.1f}g")
    
    practical_pct = (sum(all_practical) / len(all_practical)) * 100
    safe_pct = (sum(all_safe) / len(all_safe)) * 100
    
    print(f"\nPracticality:")
    print(f"  Practical portions (≥{MIN_PRACTICAL_PORTION}g): {sum(all_practical)}/{len(all_practical)} ({practical_pct:.1f}%)")
    print(f"  Too small (<{MIN_PRACTICAL_PORTION}g): {len(all_practical) - sum(all_practical)}/{len(all_practical)} ({100-practical_pct:.1f}%)")
    
    print(f"\nSafety:")
    print(f"  Safe portions: {sum(all_safe)}/{len(all_safe)} ({safe_pct:.1f}%)")
    print(f"  Unsafe portions: {len(all_safe) - sum(all_safe)}/{len(all_safe)} ({100-safe_pct:.1f}%)")
    
    print(f"\nLimiting factors:")
    total_limiting = sum(limiting_factors.values())
    for factor, count in limiting_factors.items():
        pct = (count / total_limiting) * 100 if total_limiting > 0 else 0
        print(f"  {factor}: {count}/{total_limiting} ({pct:.1f}%)")

# Statistics by algorithm
print("\nStatistics by algorithm:")

for algorithm in recommendations_with_portions:
    algo_portions = []
    algo_practical = []
    algo_safe = []
    
    for user_name, recs in recommendations_with_portions[algorithm].items():
        for rec in recs:
            algo_portions.append(rec['portion_g'])
            algo_practical.append(rec['is_practical'])
            algo_safe.append(rec['is_safe'])
    
    if algo_portions:
        practical_pct = (sum(algo_practical) / len(algo_practical)) * 100
        safe_pct = (sum(algo_safe) / len(algo_safe)) * 100
        
        print(f"\n{algorithm.upper()}:")
        print(f"  Total recommendations: {len(algo_portions)}")
        print(f"  Mean portion: {np.mean(algo_portions):.1f}g")
        print(f"  Practical: {sum(algo_practical)}/{len(algo_practical)} ({practical_pct:.1f}%)")
        print(f"  Safe: {sum(algo_safe)}/{len(algo_safe)} ({safe_pct:.1f}%)")


# ============================================================
# STEP 3.5: SAVE RECOMMENDATIONS WITH PORTIONS
# ============================================================

print("\nSTEP 3.5: SAVING RECOMMENDATIONS WITH PORTIONS")

# Save to pickle file
with open('recommendations_with_portions.pkl', 'wb') as f:
    pickle.dump(recommendations_with_portions, f)

print(f"\nSaved: recommendations_with_portions.pkl")

# Show sample for first user
sample_algorithm = 'hybrid'
if sample_algorithm in recommendations_with_portions:
    sample_users = list(recommendations_with_portions[sample_algorithm].keys())
    if sample_users:
        sample_user = sample_users[0]
        sample_recs = recommendations_with_portions[sample_algorithm][sample_user]
        
        print(f"Sample recommendations with portions:")
        print(f"User: {sample_user}, Algorithm: {sample_algorithm}")
        
        for i, rec in enumerate(sample_recs[:5], 1):
            practical = "✓" if rec['is_practical'] else "✗"
            safe = "✓" if rec['is_safe'] else "✗"
            print(f"\n{i}. {rec['food'][:50]}...")
            print(f"   Score: {rec['score']:.3f}")
            print(f"   Safe portion: {rec['portion_g']:.0f}g {practical}")
            print(f"   PHE: {rec['phe_mg']:.1f}mg | Protein: {rec['protein_g']:.1f}g | Energy: {rec['energy_kcal']:.0f}kcal")
            print(f"   Limiting factor: {rec['limiting_factor']} | Safe: {safe}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nSTAGE 3 COMPLETE - SUMMARY")

print(f"""
Files Created:
  1. recommendations_with_portions.pkl - All recommendations with safe portions

Portion Calculations:
  • Total recommendations processed: {processed}
  • Mean safe portion: {np.mean(all_portions):.1f}g
  • Practical portions (≥{MIN_PRACTICAL_PORTION}g): {practical_pct:.1f}%
  • Safe portions: {safe_pct:.1f}%
  • Main limiting factor: {max(limiting_factors, key=limiting_factors.get)}
  
KEY FIX APPLIED:
  ✓ PHE minimum relaxed to {PHE_MIN_RELAXATION*100:.0f}% for individual meals
  ✓ This should reduce infeasible recommendations from 36% to ~10-15%

Key Insights:
  • Each recommendation now has:
    - Safe portion size (grams)
    - Exact nutrients at that portion (PHE, protein, calories)
    - Practicality flag (is it a real meal or too small?)
    - Safety flag (meets all nutritional constraints)
    - Which nutrient is limiting (PHE or PROTEIN)
  
  • Portions calculated assuming ONE MEAL = {MEAL_FRACTION*100:.0f}% of daily limit
  • PHE minimum requirement: {PHE_MIN_RELAXATION*100:.0f}% (recognizes multi-meal distribution)
  • Portions <{MIN_PRACTICAL_PORTION}g marked as impractical (too small to be useful)
  • Safety checks: PHE range, protein ceiling, practical size

Next Steps:
  Stage 4: Evaluate Preference Alignment
  - Check: Do users LIKE the recommended foods?
  - Metrics: Precision@10, Recall@10, F1@10, NDCG@10
""")