import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pku.paths import paths

import pandas as pd
import numpy as np
import os
import requests
import json
from pathlib import Path
import toml

from pku.nutrition import get_child_adult_daily_needs, impute_phenylalanine

 
print("DATA PREPARATION")
 

 # CONFIGURATION & CACHING
 
USDA_API_KEY = os.environ.get("USDA_API_KEY") or None
if not USDA_API_KEY and paths.secrets_file.exists():
    secrets = toml.load(str(paths.secrets_file))
    USDA_API_KEY = secrets.get("USDA_API_KEY") or None
if USDA_API_KEY:
    print("USDA API key loaded")
else:
    print("No USDA API key. Local nutrition tables will be used.")

# Cache for API results
CACHE_FILE = Path(str(paths.usda_cache))
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


 # USDA API FUNCTIONS
 
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
        
        phe, imputed = impute_phenylalanine(nutrients['phe(mg)'], nutrients['protein(g)'])
        nutrients['phe(mg)'] = phe
        nutrients['phe_imputed'] = imputed
        
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


 # CSV FALLBACK FUNCTIONS
 
def load_csv_databases():
    """Load BOTH CSV files for fallback"""
    databases = []
    
    # Try consolidated_chat_ingredients.csv
    try:
        df = pd.read_csv(str(paths.chat_ingredients))
        
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
            databases.append((str(paths.chat_ingredients), df))
            print(f"✓ Loaded consolidated_chat_ingredients.csv: {len(df)} ingredients")
    except Exception as e:
        print(f"⚠ Could not load consolidated_chat_ingredients.csv: {e}")
    
    # Try Nutritional_Data.csv
    try:
        df = pd.read_csv(str(paths.nutrition_db))
        
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
            databases.append((str(paths.nutrition_db), df))
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
    any_imputed = False
    
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
        if match.get("phe_imputed"):
            any_imputed = True
    
    return {
        "phe_mg_total": total_phe,
        "protein_g_total": total_protein,
        "calories_total": total_calories,
        "weight_g_total": total_weight,
        "ingredients_found": ingredients_found,
        "ingredients_missing": ingredients_missing,
        "phe_imputed": any_imputed,
    }


 # STEP 1: LOAD USER RATINGS
 
 
print("STEP 1: LOADING USER RATINGS")
 

ratings_file = str(paths.fixture_ratings)

if not os.path.exists(ratings_file):
    print(f"❌ ERROR: {ratings_file} not found!")
    exit()

ratings_df = pd.read_csv(ratings_file)
required = {"user_id", "age_years", "sex", "weight_kg", "food", "rating"}
missing = required - set(ratings_df.columns)
if missing:
    print(f"ERROR: fixture is missing columns: {sorted(missing)}")
    sys.exit(1)

user_food_df = ratings_df[["user_id", "food", "rating"]].copy()
user_food_df["rating"] = pd.to_numeric(user_food_df["rating"], errors="coerce")
user_food_df = user_food_df.dropna(subset=["rating"])
user_food_df = user_food_df.drop_duplicates(subset=["user_id", "food"], keep="first")
user_food_df.to_csv(str(paths.user_ratings), index=False)
print(f"Saved {len(user_food_df)} ratings for {user_food_df['user_id'].nunique()} demo users")


 # STEP 2: LOAD CUISINE FILES
 
 
print("STEP 2: LOADING CUISINE FILES")
 

cuisine_files = {
    'African Foods': str(paths.cuisines / str(paths.cuisines / "African_Foods.csv")),
    'Central European Foods': str(paths.cuisines / str(paths.cuisines / "Central_European_Foods.csv")),
    'Chinese Foods': str(paths.cuisines / str(paths.cuisines / "Chinese_Foods.csv")),
    'Eastern European Foods': str(paths.cuisines / str(paths.cuisines / "Eastern_European_Foods.csv")),
    'Indian Foods': str(paths.cuisines / str(paths.cuisines / "Indian_Foods.csv")),
    'Italian Foods': str(paths.cuisines / str(paths.cuisines / "Italian_Foods.csv")),
    'Japanese Foods': str(paths.cuisines / str(paths.cuisines / "Japanese_Foods.csv")),
    'Mediterranean Foods': str(paths.cuisines / str(paths.cuisines / "Mediterranean_Foods.csv")),
    'Mexican Foods': str(paths.cuisines / str(paths.cuisines / "Mexican_Foods.csv")),
    'Scottish-Irish Foods': str(paths.cuisines / str(paths.cuisines / "Scottish-Irish_Foods.csv"))
}

meal_ingredients_list = []

for cuisine_name, filename in cuisine_files.items():
    if not os.path.exists(filename):
        print(f"  ⚠ {filename} not found")
        continue
    
    print(f"  Processing {cuisine_name}...")
    
    try:
        df = pd.read_csv(filename, encoding="utf-8")
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

meal_ingredients_df.to_csv(str(paths.meal_ingredients), index=False)
print(f"✓ Saved: data_meal_ingredients.csv")


 # STEP 3: LOAD CSV DATABASES
 
 
print("STEP 3: LOADING CSV DATABASES")
 

csv_databases = load_csv_databases()

refresh_nutrition = "--refresh-nutrition" in sys.argv
if not csv_databases and (refresh_nutrition or not paths.food_db.exists()):
    print("No local nutrition tables loaded.")
    if not USDA_API_KEY:
        print("No USDA API key and no food database. Cannot build meal nutrition.")
        sys.exit(1)


 # STEP 4: CALCULATE MEAL NUTRITION
 
print("STEP 4: MEAL NUTRITION")

if paths.food_db.exists() and not refresh_nutrition:
    print(f"Reusing {paths.food_db.name}. Pass --refresh-nutrition to recompute.")
    meal_nutrition_df = pd.read_csv(paths.food_db)
    meals_complete = int((meal_nutrition_df.get("data_quality") == "complete").sum()) if "data_quality" in meal_nutrition_df else 0
    meals_partial = int((meal_nutrition_df.get("data_quality") == "partial").sum()) if "data_quality" in meal_nutrition_df else 0
    meals_missing = int((meal_nutrition_df.get("data_quality") == "missing").sum()) if "data_quality" in meal_nutrition_df else 0
    meal_nutrition_list = meal_nutrition_df.to_dict("records")
else:
    meal_nutrition_list = []
    meals_complete = 0
    meals_partial = 0
    meals_missing = 0

if refresh_nutrition or not paths.food_db.exists():
    for idx, row in meal_ingredients_df.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(meal_ingredients_df)}")

        meal_name = row["full_name"]
        ingredients = row["ingredients"].split("|")
        weights = []
        for raw_weight in row["weights_g"].split("|"):
            try:
                weights.append(float(raw_weight))
            except ValueError:
                weights.append(100.0)

        result = compute_dish_nutrients(ingredients, weights, csv_databases)
        total_weight = result["weight_g_total"]
        if total_weight > 0:
            phe_per_100g = (result["phe_mg_total"] / total_weight) * 100
            protein_per_100g = (result["protein_g_total"] / total_weight) * 100
            energy_per_100g = (result["calories_total"] / total_weight) * 100
        else:
            phe_per_100g = 0.0
            protein_per_100g = 0.0
            energy_per_100g = 0.0

        if result["ingredients_missing"] == 0:
            meals_complete += 1
            data_quality = "complete"
        elif result["ingredients_found"] > 0:
            meals_partial += 1
            data_quality = "partial"
        else:
            meals_missing += 1
            data_quality = "missing"

        if result.get("phe_imputed"):
            phe_source = "imputed_50mg_per_g_protein"
        elif data_quality == "missing":
            phe_source = "missing"
        else:
            phe_source = "table_or_usda"

        meal_nutrition_list.append({
            "food_name": meal_name,
            "phe_mg_per_100g": phe_per_100g,
            "protein_g_per_100g": protein_per_100g,
            "energy_kcal_per_100g": energy_per_100g,
            "serving_size_g": total_weight,
            "ingredients_found": result["ingredients_found"],
            "ingredients_missing": result["ingredients_missing"],
            "data_quality": data_quality,
            "phe_source": phe_source,
        })

    save_cache()
    meal_nutrition_df = pd.DataFrame(meal_nutrition_list)
    meal_nutrition_df.to_csv(str(paths.food_db), index=False)
    meals_with_issues = meal_nutrition_df[meal_nutrition_df["data_quality"] != "complete"]
    if len(meals_with_issues) > 0:
        meals_with_issues.to_csv(str(paths.missing_foods), index=False)

print(f"Foods: {len(meal_nutrition_df)} (complete {meals_complete}, partial {meals_partial}, missing {meals_missing})")


 # STEP 5: USER PROFILES
 
 
print("STEP 5: USER PROFILES")
 

user_profiles = []
people = ratings_df.drop_duplicates(subset=["user_id"])
for _, row in people.iterrows():
    age_years = float(row["age_years"])
    weight_kg = float(row["weight_kg"])
    sex = row["sex"]
    limits = get_child_adult_daily_needs(age_years * 12, weight_kg, sex)
    limits.update({
        "user_id": row["user_id"],
        "age_years": age_years,
        "weight_kg": weight_kg,
        "sex": sex,
    })
    user_profiles.append(limits)

user_limits_df = pd.DataFrame(user_profiles)

print(f"✓ Calculated limits for {len(user_limits_df)} users")

user_limits_df.to_csv(str(paths.user_limits), index=False)
print(f"✓ Saved: data_user_nutritional_limits.csv")


 # FINAL SUMMARY
 
 
print("STAGE 0 COMPLETE")
 

print(f"""
Food table rows: {len(meal_nutrition_df)}
Demo users: {user_limits_df['user_id'].nunique()}
Next: python pipeline/cluster_meals.py
""")

 