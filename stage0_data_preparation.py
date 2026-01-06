import pandas as pd
import numpy as np
import os
import requests
import streamlit as st
from datetime import datetime


print("STAGE 0: DATA PREPARATION FOR PKU RECOMMENDER EVALUATION (COMPLETE)")


# ============================================================
# STEP 0.0: CHECKING USDA API KEY
# ============================================================

print("\nSTEP 0.0: CHECKING USDA API KEY")

USDA_API_KEY = None
try:
    USDA_API_KEY = st.secrets["USDA_API_KEY"]
    print("✓ Found USDA API key in Streamlit secrets")
except:
    USDA_API_KEY = os.environ.get('USDA_API_KEY')
    if USDA_API_KEY:
        print("✓ Found USDA API key in environment variable")

if not USDA_API_KEY:
    print("⚠ NO USDA API KEY FOUND!")
    print("  Get one free at: https://fdc.nal.usda.gov/api-key-signup.html")
    print("  Continuing with CSV fallback only...")
else:
    print(f"✓ USDA API key available: {USDA_API_KEY[:8]}...")


# ============================================================
# USDA API FUNCTIONS
# ============================================================

def search_usda_foods(query):
    """Search USDA FoodData Central"""
    if not query or USDA_API_KEY is None:
        return []
    
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": query,
        "pageSize": 20,
        "api_key": USDA_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('foods', [])
    except Exception as e:
        print(f"  USDA API error for '{query}': {e}")
        return []

def get_usda_food_details(fdc_id):
    """Get detailed nutrition for a specific food"""
    if USDA_API_KEY is None:
        return None
        
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    params = {"api_key": USDA_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        nutrients = {
            "phe_mg_per_100g": 0.0,
            "protein_g_per_100g": 0.0,
            "energy_kcal_per_100g": 0.0,
            "usda_name": data.get('description', '').strip().lower(),
            "source": "USDA"
        }
        
        for nutrient in data.get('foodNutrients', []):
            nutrient_name = nutrient.get('nutrient', {}).get('name', '').lower()
            amount = nutrient.get('amount', 0)
            
            if 'phenylalanine' in nutrient_name:
                nutrients['phe_mg_per_100g'] = float(amount)
            elif nutrient_name == 'protein':
                nutrients['protein_g_per_100g'] = float(amount)
            elif nutrient_name == 'energy':
                unit = nutrient.get('nutrient', {}).get('unitName', '').lower()
                if 'kj' in unit:
                    nutrients['energy_kcal_per_100g'] = float(amount) * 0.239006
                else:
                    nutrients['energy_kcal_per_100g'] = float(amount)
        
        # If PHE not provided, estimate from protein (50 mg per gram)
        if nutrients['phe_mg_per_100g'] == 0.0 and nutrients['protein_g_per_100g'] > 0:
            nutrients['phe_mg_per_100g'] = nutrients['protein_g_per_100g'] * 50.0
            nutrients['phe_estimated'] = True
        else:
            nutrients['phe_estimated'] = False
        
        return nutrients
    except Exception as e:
        print(f"  Error fetching details for ID {fdc_id}: {e}")
        return None

def search_usda_fallback(ingredient_name):
    """Search USDA for ingredient nutrition"""
    search_term = ingredient_name.lower().strip()
    
    # Improved search terms for common ingredients
    search_improvements = {
        'mint': 'spearmint fresh',
        'basil': 'basil fresh',
        'parsley': 'parsley fresh',
        'cilantro': 'coriander leaves',
        'coriander': 'coriander leaves',
        'olive oil': 'oil olive',
        'vegetable oil': 'oil vegetable',
        'salt': 'salt table',
        'pepper': 'pepper black',
        'lemon': 'lemon raw',
        'lime': 'lime raw',
        'garlic': 'garlic raw',
        'onion': 'onion raw',
        'onions': 'onion raw',
        'tomato': 'tomato red ripe raw',
        'tomatoes': 'tomato red ripe raw',
        'ginger': 'ginger root raw',
        'potatoes': 'potato raw',
        'carrots': 'carrot raw',
        'cucumber': 'cucumber raw'
    }
    
    for key, improved in search_improvements.items():
        if key in search_term:
            search_term = improved
            break
    
    usda_results = search_usda_foods(search_term)
    if not usda_results:
        return None
    
    # Prioritize Foundation and SR Legacy foods
    exclude_keywords = ['candy', 'chocolate', 'cookie', 'cake', 'ice cream',
                       'egg', 'chicken', 'beef', 'pork', 'fish', 'cheese', 
                       'milk', 'yogurt', 'supplement', 'formula']
    
    priority_results = []
    acceptable_results = []
    
    for result in usda_results[:10]:
        description = result.get('description', '').lower()
        data_type = result.get('dataType', '')
        
        if any(keyword in description for keyword in exclude_keywords):
            continue
            
        if data_type in ['Foundation', 'SR Legacy']:
            priority_results.append(result)
        elif data_type in ['Survey (FNDDS)']:
            acceptable_results.append(result)
    
    for result in (priority_results + acceptable_results):
        fdc_id = result.get('fdcId')
        if fdc_id:
            details = get_usda_food_details(fdc_id)
            if details:
                details['usda_match'] = result.get('description', 'Unknown')
                details['usda_data_type'] = result.get('dataType', 'Unknown')
                return details
    
    return None


# ============================================================
# CSV FALLBACK FUNCTIONS
# ============================================================

def normalize_name(s):
    """Normalize ingredient name for matching"""
    return str(s).strip().lower()

def load_csv_databases():
    """Load consolidated and nutritional CSV databases"""
    databases = []
    
    # Load consolidated_chat_ingredients.csv
    try:
        df = pd.read_csv("consolidated_chat_ingredients.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Rename columns to standard format
        rename_map = {}
        for col in df.columns:
            if 'ingredient' in col:
                rename_map[col] = 'name'
            elif 'phe' in col and 'mg' in col:
                rename_map[col] = 'phe_mg_per_100g'
            elif 'protein' in col and 'g' in col:
                rename_map[col] = 'protein_g_per_100g'
            elif 'energy' in col or 'kcal' in col or 'calor' in col:
                rename_map[col] = 'energy_kcal_per_100g'
            elif 'serving' in col and 'size' in col:
                rename_map[col] = 'serving_size_g'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Standardize column names
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str).str.strip().str.lower()
        
        # Convert numeric columns
        for col in ['phe_mg_per_100g', 'protein_g_per_100g', 'energy_kcal_per_100g', 'serving_size_g']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Scale to per 100g if serving size is provided
        if 'serving_size_g' in df.columns and 'phe_mg_per_100g' in df.columns:
            mask = df['serving_size_g'] > 0
            df.loc[mask, 'phe_mg_per_100g'] = (df.loc[mask, 'phe_mg_per_100g'] / df.loc[mask, 'serving_size_g']) * 100
            df.loc[mask, 'protein_g_per_100g'] = (df.loc[mask, 'protein_g_per_100g'] / df.loc[mask, 'serving_size_g']) * 100
            df.loc[mask, 'energy_kcal_per_100g'] = (df.loc[mask, 'energy_kcal_per_100g'] / df.loc[mask, 'serving_size_g']) * 100
        
        databases.append(('consolidated_chat_ingredients.csv', df))
        print(f"✓ Loaded consolidated_chat_ingredients.csv: {len(df)} ingredients")
    except Exception as e:
        print(f"⚠ Could not load consolidated_chat_ingredients.csv: {e}")
    
    # Load Nutritional_Data.csv
    try:
        df = pd.read_csv("Nutritional_Data.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Rename columns to standard format
        rename_map = {}
        for col in df.columns:
            if 'ingredient' in col:
                rename_map[col] = 'name'
            elif 'phe' in col and '(' not in col:
                rename_map[col] = 'phe_mg'
            elif 'protein' in col and '(' not in col:
                rename_map[col] = 'protein_g'
            elif 'energy' in col or 'kcal' in col:
                rename_map[col] = 'energy_kcal'
            elif 'serving' in col and 'size' in col:
                rename_map[col] = 'serving_size_g'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Standardize column names
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str).str.strip().str.lower()
        
        # Convert numeric columns
        for col in ['phe_mg', 'protein_g', 'energy_kcal', 'serving_size_g']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Scale to per 100g
        if 'serving_size_g' in df.columns:
            mask = df['serving_size_g'] > 0
            df['phe_mg_per_100g'] = 0.0
            df['protein_g_per_100g'] = 0.0
            df['energy_kcal_per_100g'] = 0.0
            
            if 'phe_mg' in df.columns:
                df.loc[mask, 'phe_mg_per_100g'] = (df.loc[mask, 'phe_mg'] / df.loc[mask, 'serving_size_g']) * 100
            if 'protein_g' in df.columns:
                df.loc[mask, 'protein_g_per_100g'] = (df.loc[mask, 'protein_g'] / df.loc[mask, 'serving_size_g']) * 100
            if 'energy_kcal' in df.columns:
                df.loc[mask, 'energy_kcal_per_100g'] = (df.loc[mask, 'energy_kcal'] / df.loc[mask, 'serving_size_g']) * 100
        
        databases.append(('Nutritional_Data.csv', df))
        print(f"✓ Loaded Nutritional_Data.csv: {len(df)} ingredients")
    except Exception as e:
        print(f"⚠ Could not load Nutritional_Data.csv: {e}")
    
    return databases

def search_csv_fallback(ingredient_name, csv_databases):
    """Search CSV databases for ingredient nutrition"""
    if not csv_databases:
        return None
    
    target = normalize_name(ingredient_name)
    
    # Try each database in order
    for db_name, df in csv_databases:
        if df.empty or 'name' not in df.columns:
            continue
        
        names_col = df['name']
        
        # Try exact match
        exact = df[names_col == target]
        if len(exact) >= 1:
            # Prefer non-zero values
            non_zero = exact[(exact.get('phe_mg_per_100g', 0) > 0) | 
                            (exact.get('protein_g_per_100g', 0) > 0) | 
                            (exact.get('energy_kcal_per_100g', 0) > 0)]
            match = non_zero.iloc[0] if not non_zero.empty else exact.iloc[0]
            
            result = {
                'phe_mg_per_100g': match.get('phe_mg_per_100g', 0.0),
                'protein_g_per_100g': match.get('protein_g_per_100g', 0.0),
                'energy_kcal_per_100g': match.get('energy_kcal_per_100g', 0.0),
                'usda_name': match.get('name', ingredient_name),
                'source': db_name,
                'match_type': 'exact'
            }
            
            # Estimate PHE if missing
            if result['phe_mg_per_100g'] == 0.0 and result['protein_g_per_100g'] > 0:
                result['phe_mg_per_100g'] = result['protein_g_per_100g'] * 50.0
                result['phe_estimated'] = True
            else:
                result['phe_estimated'] = False
            
            return result
        
        # Try startswith
        sw = df[names_col.str.startswith(target, na=False)]
        if len(sw) >= 1:
            non_zero = sw[(sw.get('phe_mg_per_100g', 0) > 0) | 
                         (sw.get('protein_g_per_100g', 0) > 0) | 
                         (sw.get('energy_kcal_per_100g', 0) > 0)]
            match = non_zero.iloc[0] if not non_zero.empty else sw.iloc[0]
            
            result = {
                'phe_mg_per_100g': match.get('phe_mg_per_100g', 0.0),
                'protein_g_per_100g': match.get('protein_g_per_100g', 0.0),
                'energy_kcal_per_100g': match.get('energy_kcal_per_100g', 0.0),
                'usda_name': match.get('name', ingredient_name),
                'source': db_name,
                'match_type': 'startswith'
            }
            
            if result['phe_mg_per_100g'] == 0.0 and result['protein_g_per_100g'] > 0:
                result['phe_mg_per_100g'] = result['protein_g_per_100g'] * 50.0
                result['phe_estimated'] = True
            else:
                result['phe_estimated'] = False
            
            return result
        
        # Try contains
        ct = df[names_col.str.contains(target, na=False, regex=False)]
        if len(ct) >= 1:
            non_zero = ct[(ct.get('phe_mg_per_100g', 0) > 0) | 
                         (ct.get('protein_g_per_100g', 0) > 0) | 
                         (ct.get('energy_kcal_per_100g', 0) > 0)]
            match = non_zero.iloc[0] if not non_zero.empty else ct.iloc[0]
            
            result = {
                'phe_mg_per_100g': match.get('phe_mg_per_100g', 0.0),
                'protein_g_per_100g': match.get('protein_g_per_100g', 0.0),
                'energy_kcal_per_100g': match.get('energy_kcal_per_100g', 0.0),
                'usda_name': match.get('name', ingredient_name),
                'source': db_name,
                'match_type': 'contains'
            }
            
            if result['phe_mg_per_100g'] == 0.0 and result['protein_g_per_100g'] > 0:
                result['phe_mg_per_100g'] = result['protein_g_per_100g'] * 50.0
                result['phe_estimated'] = True
            else:
                result['phe_estimated'] = False
            
            return result
    
    return None


# ============================================================
# STEP 0.1: LOAD AND ANALYZE RATINGS DATA
# ============================================================

print("\n" + "="*70)
print("STEP 0.1: LOADING USER RATINGS DATA")
print("="*70)

ratings_file = "ratingsappdata - Sheet1.csv"

if not os.path.exists(ratings_file):
    print(f"❌ ERROR: File '{ratings_file}' not found!")
    exit()

ratings_df = pd.read_csv(ratings_file)

print(f"✓ Loaded {ratings_file}")
print(f"  Total rows: {len(ratings_df)}")
print(f"  Total columns: {len(ratings_df.columns)}")
print(f"  Total users: {ratings_df['Name'].nunique()}")

# Extract food ratings
demographic_cols = [
    'Name', 'Email', 'Age', 'Gender', 'Height_cm', 'Weight_kg', 
    'PHE_tolerance', 'Dietary_tolerance_mg_per_day',
    'Selected_Cuisines', 'Timestamp'
]

food_columns = []
for col in ratings_df.columns:
    if col not in demographic_cols and '(' in col and ')' in col:
        food_columns.append(col)

print(f"✓ Found {len(food_columns)} food rating columns")

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
print(f"  Unique users: {user_food_df['user_name'].nunique()}")
print(f"  Unique foods: {user_food_df['food'].nunique()}")

user_food_df.to_csv('data_user_food_ratings.csv', index=False)
print(f"✓ Saved: data_user_food_ratings.csv")


# ============================================================
# STEP 0.2: LOAD MEAL-INGREDIENT MAPPINGS FROM CUISINE CSVs
# ============================================================

print("\n" + "="*70)
print("STEP 0.2: LOADING MEAL-INGREDIENT MAPPINGS FROM CUISINE CSVs")
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
        print(f"  ⚠ {filename} not found, skipping...")
        continue
    
    print(f"  Processing {cuisine_name}...")
    
    try:
        df = pd.read_csv(filename, encoding='latin1')
        
        current_meal = None
        current_ingredients = []
        current_weights = []
        
        for idx, row in df.iterrows():
            meal_name = row.get('Meal', None)
            ingredient = row.get('Ingredient', None)
            weight = row.get('Weight (g)', None)
            
            # New meal starts
            if pd.notna(meal_name) and meal_name.strip() != '':
                # Save previous meal if exists
                if current_meal and current_ingredients:
                    meal_ingredients_list.append({
                        'cuisine': cuisine_name,
                        'meal': current_meal,
                        'full_name': f"{current_meal} ({cuisine_name})",
                        'ingredients': '|'.join(current_ingredients),
                        'weights_g': '|'.join(map(str, current_weights))
                    })
                
                # Start new meal
                current_meal = meal_name.strip()
                current_ingredients = []
                current_weights = []
            
            # Add ingredient to current meal
            if pd.notna(ingredient) and ingredient.strip() != '' and current_meal:
                current_ingredients.append(ingredient.strip())
                # Handle weight (might be missing)
                if pd.notna(weight):
                    try:
                        current_weights.append(float(weight))
                    except:
                        current_weights.append(100.0)  # Default
                else:
                    current_weights.append(100.0)
        
        # Save last meal
        if current_meal and current_ingredients:
            meal_ingredients_list.append({
                'cuisine': cuisine_name,
                'meal': current_meal,
                'full_name': f"{current_meal} ({cuisine_name})",
                'ingredients': '|'.join(current_ingredients),
                'weights_g': '|'.join(map(str, current_weights))
            })
        
        print(f"  ✓ Loaded {sum(1 for m in meal_ingredients_list if m['cuisine'] == cuisine_name)} meals")
    
    except Exception as e:
        print(f"  ❌ Error loading {filename}: {e}")

meal_ingredients_df = pd.DataFrame(meal_ingredients_list)

print(f"✓ Total meals with ingredients: {len(meal_ingredients_df)}")
print(f"✓ Cuisines loaded: {meal_ingredients_df['cuisine'].nunique()}")

meal_ingredients_df.to_csv('data_meal_ingredients.csv', index=False)
print(f"✓ Saved: data_meal_ingredients.csv")


# ============================================================
# STEP 0.3: EXTRACT ALL UNIQUE INGREDIENTS
# ============================================================

print("\n" + "="*70)
print("STEP 0.3: EXTRACTING UNIQUE INGREDIENTS")
print("="*70)

all_ingredients = set()

for _, row in meal_ingredients_df.iterrows():
    ingredients = row['ingredients'].split('|')
    all_ingredients.update([ing.lower().strip() for ing in ingredients])

print(f"✓ Found {len(all_ingredients)} unique ingredients across all cuisines")
print(f"  Sample ingredients:")
for ing in list(sorted(all_ingredients))[:10]:
    print(f"    - {ing}")


# ============================================================
# STEP 0.4: LOAD CSV DATABASES
# ============================================================

print("\n" + "="*70)
print("STEP 0.4: LOADING CSV FALLBACK DATABASES")
print("="*70)

csv_databases = load_csv_databases()

if not csv_databases:
    print("⚠ No CSV databases loaded - will rely only on USDA API")


# ============================================================
# STEP 0.5: FETCH NUTRITIONAL DATA FOR INGREDIENTS
# ============================================================

print("\n" + "="*70)
print("STEP 0.5: FETCHING NUTRITIONAL DATA FOR INGREDIENTS")
print("="*70)

print(f"Fetching nutrition for {len(all_ingredients)} ingredients...")
print("This may take several minutes...")
print()

ingredient_db = {}
found_usda = 0
found_csv = 0
missing = []

for i, ingredient in enumerate(sorted(all_ingredients), 1):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(all_ingredients)} (USDA: {found_usda}, CSV: {found_csv}, Missing: {len(missing)})...")
    
    # Try USDA first
    nutrients = None
    if USDA_API_KEY is not None:
        nutrients = search_usda_fallback(ingredient)
    
    if nutrients:
        ingredient_db[ingredient] = nutrients
        found_usda += 1
    else:
        # Try CSV fallback
        nutrients = search_csv_fallback(ingredient, csv_databases)
        
        if nutrients:
            ingredient_db[ingredient] = nutrients
            found_csv += 1
        else:
            missing.append(ingredient)

print()
print("="*70)
print("NUTRITIONAL DATA COLLECTION COMPLETE")
print("="*70)
print(f"  ✓ Found via USDA API: {found_usda}/{len(all_ingredients)} ({found_usda/len(all_ingredients)*100:.1f}%)")
print(f"  ✓ Found via CSV fallback: {found_csv}/{len(all_ingredients)} ({found_csv/len(all_ingredients)*100:.1f}%)")
print(f"  ✓ Total found: {len(ingredient_db)}/{len(all_ingredients)} ({len(ingredient_db)/len(all_ingredients)*100:.1f}%)")
print(f"  ⚠ Missing: {len(missing)} ingredients ({len(missing)/len(all_ingredients)*100:.1f}%)")

if missing:
    print(f"\n  Sample missing ingredients (first 20):")
    for ing in missing[:20]:
        print(f"    - {ing}")

# Save ingredient database
if ingredient_db:
    ingredient_db_df = pd.DataFrame([
        {'ingredient_name': name, **nutrients}
        for name, nutrients in ingredient_db.items()
    ])
    ingredient_db_df.to_csv('data_ingredient_database.csv', index=False)
    print(f"\n✓ Saved: data_ingredient_database.csv")

# Save missing ingredients list for manual lookup
if missing:
    missing_df = pd.DataFrame({'ingredient': missing})
    missing_df.to_csv('data_missing_ingredients.csv', index=False)
    print(f"✓ Saved: data_missing_ingredients.csv (for manual lookup)")


# ============================================================
# STEP 0.6: EXTRACT USER PROFILES AND CALCULATE LIMITS
# ============================================================

print("\n" + "="*70)
print("STEP 0.6: USER PROFILES & NUTRITIONAL LIMITS")
print("="*70)

def get_child_adult_daily_needs(age_months, weight_kg, sex):
    needs = {}
    if age_months < 48:
        needs.update({
            'phe_mg_min': 200, 'phe_mg_max': 400, 
            'protein_g': 30, 'energy_kcal': 1300,
            'age_group': '1-4 years'
        })
    elif age_months < 84:
        needs.update({
            'phe_mg_min': 210, 'phe_mg_max': 450,
            'protein_g': 35, 'energy_kcal': 1700,
            'age_group': '4-7 years'
        })
    elif age_months < 132:
        needs.update({
            'phe_mg_min': 220, 'phe_mg_max': 500,
            'protein_g': 40, 'energy_kcal': 2400,
            'age_group': '7-11 years'
        })
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
print(f"\n  Sample user profiles:")
print(user_limits_df[['user_name', 'age_group', 'phe_mg_max', 'protein_g', 'energy_kcal']].head().to_string(index=False))

user_limits_df.to_csv('data_user_nutritional_limits.csv', index=False)
print(f"\n✓ Saved: data_user_nutritional_limits.csv")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("STAGE 0 COMPLETE - DATA PREPARATION SUMMARY")
print("="*70)

print(f"""
FILES CREATED:
  1. data_user_food_ratings.csv           - {len(user_food_df)} user-food ratings
  2. data_meal_ingredients.csv            - {len(meal_ingredients_df)} meals with ingredient lists
  3. data_ingredient_database.csv         - {len(ingredient_db)} ingredients with nutrition data
  4. data_user_nutritional_limits.csv     - {len(user_limits_df)} user profiles with limits
  5. data_missing_ingredients.csv         - {len(missing)} ingredients needing manual lookup

STATISTICS:
  Users: {user_food_df['user_name'].nunique()}
  Rated meals: {user_food_df['food'].nunique()}
  Total meals in database: {len(meal_ingredients_df)}
  Unique ingredients: {len(all_ingredients)}
  
NUTRITIONAL DATA COVERAGE:
  USDA API: {found_usda} ingredients ({found_usda/len(all_ingredients)*100:.1f}%)
  CSV fallback: {found_csv} ingredients ({found_csv/len(all_ingredients)*100:.1f}%)
  Total covered: {len(ingredient_db)} ingredients ({len(ingredient_db)/len(all_ingredients)*100:.1f}%)
  Missing: {len(missing)} ingredients ({len(missing)/len(all_ingredients)*100:.1f}%)

DATA QUALITY:
  ✓ Ingredient-level nutritional database ready
  ✓ Meal-ingredient mappings complete
  ✓ User ratings extracted and validated
  ✓ User nutritional limits calculated
  {f'⚠ {len(missing)} ingredients need manual lookup' if missing else '✓ All ingredients have nutritional data'}

NEXT STEPS:
  1. Review data_missing_ingredients.csv
  2. Optionally add missing ingredients manually
  3. Run Stage 1 (train/test split)
  4. Implement Stage 2 (recommendation algorithms)
  5. Implement Stage 3 (portion calculation & safety)
  6. Implement Stage 4-6 (evaluation metrics)
  
KEY IMPROVEMENTS IN THIS VERSION:
  ✓ Three-tier fallback system: USDA API → CSV databases → Missing list
  ✓ {found_csv} additional ingredients recovered via CSV fallback
  ✓ Comprehensive source tracking (USDA vs CSV)
  ✓ Match type tracking (exact, startswith, contains)
  ✓ PHE estimation flagging for transparency
  ✓ Missing ingredients exported for manual review
""")

print("="*70)
print("Ready for Stage 1: Train/Test Split")
print("="*70)