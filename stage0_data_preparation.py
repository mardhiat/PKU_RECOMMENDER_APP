import pandas as pd
import numpy as np
import os
import requests
import json
from pathlib import Path
import toml

print("="*70)
print("STAGE 0: DATA PREPARATION (WITH CACHING)")
print("="*70)

# ============================================================
# CONFIGURATION & CACHING
# ============================================================

# Read from Streamlit secrets file
try:
    secrets = toml.load(".streamlit/secrets.toml")
    USDA_API_KEY = secrets.get("USDA_API_KEY")
    if USDA_API_KEY:
        print(f"✓ USDA API key loaded from secrets.toml: {USDA_API_KEY[:8]}...")
    else:
        print("⚠ No USDA_API_KEY in secrets.toml")
        USDA_API_KEY = None
except FileNotFoundError:
    print("⚠ .streamlit/secrets.toml not found")
    USDA_API_KEY = None

# Cache for API results
CACHE_FILE = Path("usda_cache.json")
if CACHE_FILE.exists():
    with open(CACHE_FILE) as f:
        USDA_CACHE = json.load(f)
    print(f"✓ Loaded {len(USDA_CACHE)} cached USDA results")
else:
    USDA_CACHE = {}
    print("✓ Starting fresh USDA cache")

def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(USDA_CACHE, f, indent=2)


# ============================================================
# USDA API FUNCTIONS
# ============================================================

def search_usda_foods(query):
    if not query or not USDA_API_KEY:
        return []
    
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": query,
        "pageSize": 20,
        "api_key": USDA_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=5)  # Reduced timeout
        response.raise_for_status()
        return response.json().get('foods', [])
    except Exception as e:
        return []


def get_usda_food_details(fdc_id):
    if not USDA_API_KEY:
        return None
        
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    params = {"api_key": USDA_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=3)  # Reduced timeout
        response.raise_for_status()
        data = response.json()
        
        nutrients = {
            "phe(mg)": 0.0,
            "protein(g)": 0.0,
            "energy(kcal)": 0.0,
            "serving_size(g)": 100.0,
            "name": data.get('description', '').strip().lower()
        }
        
        for nutrient in data.get('foodNutrients', []):
            nutrient_name = nutrient.get('nutrient', {}).get('name', '').lower()
            amount = nutrient.get('amount', 0)
            
            if 'phenylalanine' in nutrient_name:
                nutrients['phe(mg)'] = float(amount)
            elif nutrient_name == 'protein':
                nutrients['protein(g)'] = float(amount)
            elif nutrient_name == 'energy':
                unit = nutrient.get('nutrient', {}).get('unitName', '').lower()
                if 'kj' in unit:
                    nutrients['energy(kcal)'] = float(amount) * 0.239006
                else:
                    nutrients['energy(kcal)'] = float(amount)
        
        if nutrients['phe(mg)'] == 0.0 and nutrients['protein(g)'] > 0:
            nutrients['phe(mg)'] = nutrients['protein(g)'] * 50.0
        
        return pd.Series(nutrients)
    except Exception as e:
        return None


def search_usda_fallback(name):
    # Check cache first
    cache_key = name.lower().strip()
    if cache_key in USDA_CACHE:
        cached = USDA_CACHE[cache_key]
        if cached is None:
            return None
        return pd.Series(cached)
    
    search_term = name.lower().strip()
    
    search_improvements = {
        'mint': 'spearmint',
        'basil': 'basil fresh',
        'parsley': 'parsley fresh',
        'cilantro': 'coriander',
        'olive oil': 'oil olive',
        'vegetable oil': 'oil vegetable',
        'salt': 'salt table',
        'pepper': 'pepper black',
        'lemon': 'lemon raw without peel',
        'lime': 'lime raw',
        'garlic': 'garlic raw',
        'onion': 'onion raw',
        'tomato': 'tomato red ripe raw',
        'ginger': 'ginger root raw',
        'low-protein flour': 'wheat flour white',
        'water': 'water',
    }
    
    for key, improved in search_improvements.items():
        if key in search_term:
            search_term = improved
            break
    
    usda_results = search_usda_foods(search_term)
    if not usda_results:
        USDA_CACHE[cache_key] = None
        return None
    
    exclude_keywords = ['candy', 'chocolate', 'cookie', 'cake', 'ice cream',
                       'egg', 'chicken', 'beef', 'pork', 'fish', 'cheese', 
                       'milk', 'yogurt', 'supplement', 'formula']
    
    original_term = name.lower().strip().split()[0]
    
    priority_results = []
    acceptable_results = []
    
    for result in usda_results[:10]:
        description = result.get('description', '').lower()
        data_type = result.get('dataType', '')
        
        if original_term in ['mint', 'basil', 'parsley', 'cilantro']:
            if original_term not in description and 'spearmint' not in description:
                continue
        
        if any(keyword in description for keyword in exclude_keywords):
            continue
            
        if data_type in ['Foundation', 'SR Legacy']:
            priority_results.append(result)
        elif data_type in ['Survey (FNDDS)']:
            acceptable_results.append(result)
    
    result = None
    for res in (priority_results + acceptable_results):
        fdc_id = res.get('fdcId')
        if fdc_id:
            details = get_usda_food_details(fdc_id)
            if details is not None:
                details['usda_match'] = res.get('description', 'Unknown')
                details['usda_data_type'] = res.get('dataType', 'Unknown')
                result = details
                break
    
    # Cache result
    if result is not None:
        USDA_CACHE[cache_key] = result.to_dict()
    else:
        USDA_CACHE[cache_key] = None
    
    # Save cache periodically
    if len(USDA_CACHE) % 50 == 0:
        save_cache()
        print(f"    [Cached {len(USDA_CACHE)} ingredients]")
    
    return result


# ============================================================
# CSV FALLBACK FUNCTIONS
# ============================================================

def load_csv_databases():
    """Load BOTH CSV files for fallback"""
    databases = []
    
    # Try consolidated_chat_ingredients.csv
    try:
        df = pd.read_csv("consolidated_chat_ingredients.csv")
        
        df.columns = [str(c).strip() for c in df.columns]
        
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'ingredient' in col_lower:
                col_map[col] = 'name'
            elif 'phe' in col_lower and 'mg' in col_lower:
                col_map[col] = 'phe(mg)'
            elif 'protein' in col_lower and 'g' in col_lower:
                col_map[col] = 'protein(g)'
            elif 'energy' in col_lower or 'kcal' in col_lower:
                col_map[col] = 'energy(kcal)'
            elif 'serving' in col_lower and 'size' in col_lower:
                col_map[col] = 'serving_size(g)'
        
        df = df.rename(columns=col_map)
        
        for col in ['phe(mg)', 'protein(g)', 'energy(kcal)', 'serving_size(g)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str).str.strip().str.lower()
            databases.append(('consolidated_chat_ingredients.csv', df))
            print(f"✓ Loaded consolidated_chat_ingredients.csv: {len(df)} ingredients")
    except Exception as e:
        print(f"⚠ Could not load consolidated_chat_ingredients.csv: {e}")
    
    # Try Nutritional_Data.csv
    try:
        df = pd.read_csv("Nutritional_Data.csv")
        
        df.columns = [str(c).strip() for c in df.columns]
        
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'ingredient' in col_lower:
                col_map[col] = 'name'
            elif 'phe' in col_lower and '(' not in col_lower:
                col_map[col] = 'phe_mg'
            elif 'protein' in col_lower and '(' not in col_lower:
                col_map[col] = 'protein_g'
            elif 'energy' in col_lower or 'kcal' in col_lower:
                col_map[col] = 'energy_kcal'
            elif 'serving' in col_lower and 'size' in col_lower:
                col_map[col] = 'serving_size_g'
        
        df = df.rename(columns=col_map)
        
        for col in ['phe_mg', 'protein_g', 'energy_kcal', 'serving_size_g']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        if 'serving_size_g' in df.columns:
            mask = df['serving_size_g'] > 0
            df['phe(mg)'] = 0.0
            df['protein(g)'] = 0.0
            df['energy(kcal)'] = 0.0
            
            if 'phe_mg' in df.columns:
                df.loc[mask, 'phe(mg)'] = (df.loc[mask, 'phe_mg'] / df.loc[mask, 'serving_size_g']) * 100
            if 'protein_g' in df.columns:
                df.loc[mask, 'protein(g)'] = (df.loc[mask, 'protein_g'] / df.loc[mask, 'serving_size_g']) * 100
            if 'energy_kcal' in df.columns:
                df.loc[mask, 'energy(kcal)'] = (df.loc[mask, 'energy_kcal'] / df.loc[mask, 'serving_size_g']) * 100
            
            df['serving_size(g)'] = 100.0
        
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str).str.strip().str.lower()
            databases.append(('Nutritional_Data.csv', df))
            print(f"✓ Loaded Nutritional_Data.csv: {len(df)} ingredients")
    except Exception as e:
        print(f"⚠ Could not load Nutritional_Data.csv: {e}")
    
    return databases


def search_csv_fallback(ingredient_name, csv_databases):
    """Search CSV databases for ingredient"""
    if not csv_databases:
        return None
    
    target = ingredient_name.lower().strip()
    
    for db_name, df in csv_databases:
        if df.empty or 'name' not in df.columns:
            continue
        
        names_col = df['name']
        
        # Exact match
        exact = df[names_col == target]
        if len(exact) >= 1:
            match = exact.iloc[0]
            return pd.Series({
                'phe(mg)': match.get('phe(mg)', 0.0),
                'protein(g)': match.get('protein(g)', 0.0),
                'energy(kcal)': match.get('energy(kcal)', 0.0),
                'serving_size(g)': match.get('serving_size(g)', 100.0),
                'name': match.get('name', ingredient_name)
            })
        
        # Startswith
        sw = df[names_col.str.startswith(target, na=False)]
        if len(sw) >= 1:
            match = sw.iloc[0]
            return pd.Series({
                'phe(mg)': match.get('phe(mg)', 0.0),
                'protein(g)': match.get('protein(g)', 0.0),
                'energy(kcal)': match.get('energy(kcal)', 0.0),
                'serving_size(g)': match.get('serving_size(g)', 100.0),
                'name': match.get('name', ingredient_name)
            })
        
        # Contains
        ct = df[names_col.str.contains(target, na=False, regex=False)]
        if len(ct) >= 1:
            match = ct.iloc[0]
            return pd.Series({
                'phe(mg)': match.get('phe(mg)', 0.0),
                'protein(g)': match.get('protein(g)', 0.0),
                'energy(kcal)': match.get('energy(kcal)', 0.0),
                'serving_size(g)': match.get('serving_size(g)', 100.0),
                'name': match.get('name', ingredient_name)
            })
    
    return None


def select_best_match(name, csv_databases):
    """Try USDA first, then CSV fallback"""
    
    # Try USDA
    if USDA_API_KEY:
        usda_result = search_usda_fallback(name)
        if usda_result is not None:
            return usda_result
    
    # Try CSV
    csv_result = search_csv_fallback(name, csv_databases)
    if csv_result is not None:
        return csv_result
    
    return None


def scale_nutrients(row, weight_g):
    """Scale nutrition by weight"""
    serving_size = row.get("serving_size(g)", 100.0)
    if serving_size == 0:
        serving_size = 100.0
    
    phe_per_g = (row.get("phe(mg)", 0.0) / serving_size)
    prot_per_g = (row.get("protein(g)", 0.0) / serving_size)
    cal_per_g = (row.get("energy(kcal)", 0.0) / serving_size)
    
    return {
        "phe_mg": phe_per_g * weight_g,
        "protein_g": prot_per_g * weight_g,
        "calories": cal_per_g * weight_g,
    }


def compute_dish_nutrients(ingredients_list, weights_list, csv_databases):
    """Calculate meal nutrition from ingredients"""
    total_phe = 0.0
    total_protein = 0.0
    total_calories = 0.0
    total_weight = 0.0
    ingredients_found = 0
    ingredients_missing = 0
    
    for ing_name, weight in zip(ingredients_list, weights_list):
        if not ing_name or weight <= 0:
            continue
        
        match = select_best_match(ing_name, csv_databases)
        
        if match is None:
            ingredients_missing += 1
            total_weight += weight
            continue
        
        scaled = scale_nutrients(match, weight)
        total_phe += scaled["phe_mg"]
        total_protein += scaled["protein_g"]
        total_calories += scaled["calories"]
        total_weight += weight
        ingredients_found += 1
    
    return {
        "phe_mg_total": total_phe,
        "protein_g_total": total_protein,
        "calories_total": total_calories,
        "weight_g_total": total_weight,
        "ingredients_found": ingredients_found,
        "ingredients_missing": ingredients_missing
    }


# ============================================================
# STEP 1: LOAD USER RATINGS
# ============================================================

print("\n" + "="*70)
print("STEP 1: LOADING USER RATINGS")
print("="*70)

ratings_file = "ratingsappdata - Sheet1.csv"

if not os.path.exists(ratings_file):
    print(f"❌ ERROR: {ratings_file} not found!")
    exit()

ratings_df = pd.read_csv(ratings_file)

print(f"✓ Loaded {len(ratings_df)} rows")

demographic_cols = [
    'Name', 'Email', 'Age', 'Gender', 'Height_cm', 'Weight_kg', 
    'PHE_tolerance', 'Dietary_tolerance_mg_per_day',
    'Selected_Cuisines', 'Timestamp'
]

food_columns = [col for col in ratings_df.columns 
                if col not in demographic_cols and '(' in col and ')' in col]

print(f"✓ Found {len(food_columns)} food columns")

user_food_ratings = []

for idx, row in ratings_df.iterrows():
    user_name = row['Name']
    user_email = row['Email']
    
    for food_col in food_columns:
        rating = row[food_col]
        
        if pd.notna(rating) and rating != '':
            try:
                rating_value = float(rating)
                if 0 <= rating_value <= 5:
                    user_food_ratings.append({
                        'user_name': user_name,
                        'user_email': user_email,
                        'food': food_col,
                        'rating': rating_value
                    })
            except (ValueError, TypeError):
                continue

user_food_df = pd.DataFrame(user_food_ratings)

print(f"✓ Extracted {len(user_food_df)} ratings")

user_food_df.to_csv('data_user_food_ratings.csv', index=False)
print(f"✓ Saved: data_user_food_ratings.csv")


# ============================================================
# STEP 2: LOAD CUISINE FILES
# ============================================================

print("\n" + "="*70)
print("STEP 2: LOADING CUISINE FILES")
print("="*70)

cuisine_files = {
    'African Foods': 'African_Foods.csv',
    'Central European Foods': 'Central_European_Foods.csv',
    'Chinese Foods': 'Chinese_Foods.csv',
    'Eastern European Foods': 'Eastern_European_Foods.csv',
    'Indian Foods': 'Indian_Foods.csv',
    'Italian Foods': 'Italian_Foods.csv',
    'Japanese Foods': 'Japanese_Foods.csv',
    'Mediterranean Foods': 'Mediterranean_Foods.csv',
    'Mexican Foods': 'Mexican_Foods.csv',
    'Scottish-Irish Foods': 'Scottish-Irish_Foods.csv'
}

meal_ingredients_list = []

for cuisine_name, filename in cuisine_files.items():
    if not os.path.exists(filename):
        print(f"  ⚠ {filename} not found")
        continue
    
    print(f"  Processing {cuisine_name}...")
    
    try:
        df = pd.read_csv(filename, encoding='latin1')
        df.columns = [c.strip().lower() for c in df.columns]
        
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "meal":
                rename_map[c] = "dish"
            elif cl == "ingredient":
                rename_map[c] = "ingredient"
            elif cl in ["grams", "weight (g)", "weight"]:
                rename_map[c] = "amount"
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        if "dish" in df.columns:
            df["dish"] = df["dish"].ffill()
        
        for dish_name in df['dish'].unique():
            if pd.isna(dish_name):
                continue
            
            dish_rows = df[df['dish'] == dish_name]
            ingredients = []
            weights = []
            
            for _, row in dish_rows.iterrows():
                ing = row.get('ingredient', None)
                weight = row.get('amount', None)
                
                if pd.notna(ing) and pd.notna(weight):
                    ingredients.append(str(ing).strip())
                    try:
                        weights.append(float(weight))
                    except:
                        weights.append(100.0)
            
            if ingredients:
                meal_ingredients_list.append({
                    'cuisine': cuisine_name,
                    'meal': dish_name,
                    'full_name': f"{dish_name} ({cuisine_name})",
                    'ingredients': '|'.join(ingredients),
                    'weights_g': '|'.join(map(str, weights))
                })
        
        print(f"  ✓ Loaded {sum(1 for m in meal_ingredients_list if m['cuisine'] == cuisine_name)} meals")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")

meal_ingredients_df = pd.DataFrame(meal_ingredients_list)

print(f"✓ Total meals: {len(meal_ingredients_df)}")

meal_ingredients_df.to_csv('data_meal_ingredients.csv', index=False)
print(f"✓ Saved: data_meal_ingredients.csv")


# ============================================================
# STEP 3: LOAD CSV DATABASES
# ============================================================

print("\n" + "="*70)
print("STEP 3: LOADING CSV DATABASES")
print("="*70)

csv_databases = load_csv_databases()

if not csv_databases:
    print("❌ NO CSV DATABASES LOADED!")
    if not USDA_API_KEY:
        print("❌ NO USDA API KEY AND NO CSV FALLBACK - CANNOT PROCEED!")
        exit()


# ============================================================
# STEP 4: CALCULATE MEAL NUTRITION
# ============================================================

print("\n" + "="*70)
print("STEP 4: CALCULATING MEAL NUTRITION")
print("="*70)

meal_nutrition_list = []
meals_complete = 0
meals_partial = 0
meals_missing = 0

for idx, row in meal_ingredients_df.iterrows():
    if (idx + 1) % 10 == 0:  # Print every 10 meals
        print(f"  Progress: {idx+1}/{len(meal_ingredients_df)} ({(idx+1)/len(meal_ingredients_df)*100:.1f}%)")
    
    meal_name = row['full_name']
    ingredients = row['ingredients'].split('|')
    weights_str = row['weights_g'].split('|')
    
    weights = []
    for w in weights_str:
        try:
            weights.append(float(w))
        except:
            weights.append(100.0)
    
    result = compute_dish_nutrients(ingredients, weights, csv_databases)
    
    total_weight = result['weight_g_total']
    if total_weight > 0:
        phe_per_100g = (result['phe_mg_total'] / total_weight) * 100
        protein_per_100g = (result['protein_g_total'] / total_weight) * 100
        energy_per_100g = (result['calories_total'] / total_weight) * 100
    else:
        phe_per_100g = 0.0
        protein_per_100g = 0.0
        energy_per_100g = 0.0
    
    if result['ingredients_missing'] == 0:
        meals_complete += 1
        data_quality = "complete"
    elif result['ingredients_found'] > 0:
        meals_partial += 1
        data_quality = "partial"
    else:
        meals_missing += 1
        data_quality = "missing"
    
    meal_nutrition_list.append({
        'food_name': meal_name,
        'phe_mg_per_100g': phe_per_100g,
        'protein_g_per_100g': protein_per_100g,
        'energy_kcal_per_100g': energy_per_100g,
        'serving_size_g': total_weight,
        'ingredients_found': result['ingredients_found'],
        'ingredients_missing': result['ingredients_missing'],
        'data_quality': data_quality
    })

# Save cache one final time
save_cache()
print(f"\n✓ Final cache saved: {len(USDA_CACHE)} ingredients")

print()
print(f"✓ Processed {len(meal_nutrition_list)} meals")
print(f"  Complete data: {meals_complete} ({meals_complete/len(meal_nutrition_list)*100:.1f}%)")
print(f"  Partial data: {meals_partial} ({meals_partial/len(meal_nutrition_list)*100:.1f}%)")
print(f"  No data: {meals_missing} ({meals_missing/len(meal_nutrition_list)*100:.1f}%)")

meal_nutrition_df = pd.DataFrame(meal_nutrition_list)
meal_nutrition_df.to_csv('data_food_database.csv', index=False)
print(f"✓ Saved: data_food_database.csv")

meals_with_issues = meal_nutrition_df[meal_nutrition_df['data_quality'] != 'complete']
if len(meals_with_issues) > 0:
    meals_with_issues.to_csv('data_missing_foods.csv', index=False)
    print(f"✓ Saved: data_missing_foods.csv ({len(meals_with_issues)} meals)")


# ============================================================
# STEP 5: USER PROFILES
# ============================================================

print("\n" + "="*70)
print("STEP 5: USER PROFILES")
print("="*70)

def get_child_adult_daily_needs(age_months, weight_kg, sex):
    needs = {}
    if age_months < 48:
        needs.update({'phe_mg_min': 200, 'phe_mg_max': 400, 'protein_g': 30, 'energy_kcal': 1300, 'age_group': '1-4 years'})
    elif age_months < 84:
        needs.update({'phe_mg_min': 210, 'phe_mg_max': 450, 'protein_g': 35, 'energy_kcal': 1700, 'age_group': '4-7 years'})
    elif age_months < 132:
        needs.update({'phe_mg_min': 220, 'phe_mg_max': 500, 'protein_g': 40, 'energy_kcal': 2400, 'age_group': '7-11 years'})
    elif age_months < 180:
        if sex == "Female":
            needs.update({'phe_mg_min': 250, 'phe_mg_max': 750, 'protein_g': 50, 'energy_kcal': 2200, 'age_group': '11-15 years'})
        else:
            needs.update({'phe_mg_min': 225, 'phe_mg_max': 900, 'protein_g': 55, 'energy_kcal': 2700, 'age_group': '11-15 years'})
    elif age_months < 228:
        if sex == "Female":
            needs.update({'phe_mg_min': 230, 'phe_mg_max': 700, 'protein_g': 55, 'energy_kcal': 2100, 'age_group': '15-19 years'})
        else:
            needs.update({'phe_mg_min': 295, 'phe_mg_max': 1100, 'protein_g': 65, 'energy_kcal': 2800, 'age_group': '15-19 years'})
    else:
        if sex == "Female":
            needs.update({'phe_mg_min': 220, 'phe_mg_max': 700, 'protein_g': 60, 'energy_kcal': 2100, 'age_group': '19+ years'})
        else:
            needs.update({'phe_mg_min': 290, 'phe_mg_max': 1200, 'protein_g': 70, 'energy_kcal': 2900, 'age_group': '19+ years'})
    return needs

user_profiles = []
for idx, row in ratings_df.iterrows():
    age = row.get('Age', None)
    weight_kg = row.get('Weight_kg', None)
    gender = row.get('Gender', None)
    
    if pd.isna(age) or pd.isna(weight_kg):
        continue
    
    age_months = age * 12
    limits = get_child_adult_daily_needs(age_months, weight_kg, gender)
    
    limits.update({
        'user_name': row['Name'],
        'user_email': row['Email'],
        'age': age,
        'weight_kg': weight_kg,
        'gender': gender
    })
    user_profiles.append(limits)

user_limits_df = pd.DataFrame(user_profiles)

print(f"✓ Calculated limits for {len(user_limits_df)} users")

user_limits_df.to_csv('data_user_nutritional_limits.csv', index=False)
print(f"✓ Saved: data_user_nutritional_limits.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("STAGE 0 COMPLETE")
print("="*70)

print(f"""
MEAL NUTRITION:
  Complete data: {meals_complete}/{len(meal_nutrition_df)} ({meals_complete/len(meal_nutrition_df)*100:.1f}%)
  Partial data: {meals_partial} ({meals_partial/len(meal_nutrition_df)*100:.1f}%)
  No data: {meals_missing} ({meals_missing/len(meal_nutrition_list)*100:.1f}%)

USDA CACHE:
  Total cached ingredients: {len(USDA_CACHE)}
  Cache file: usda_cache.json

NEXT: Run Stages 1-5 to see evaluation results!
""")

print("="*70)