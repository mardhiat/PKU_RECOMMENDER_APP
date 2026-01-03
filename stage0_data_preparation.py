import pandas as pd
import numpy as np
import os
import requests
import streamlit as st
from datetime import datetime


print("STAGE 0: DATA PREPARATION FOR PKU RECOMMENDER EVALUATION")



# GET USDA API KEY



print("CHECKING USDA API KEY")


# Try to get API key from Streamlit secrets first
USDA_API_KEY = None
try:
    USDA_API_KEY = st.secrets["USDA_API_KEY"]
    print("Found USDA API key in Streamlit secrets")
except:
    print("⚠ No Streamlit secrets found, trying environment variable...")
    USDA_API_KEY = os.environ.get('USDA_API_KEY')
    if USDA_API_KEY:
        print("Found USDA API key in environment variable")

if not USDA_API_KEY:
    print("\nNO USDA API KEY FOUND!")
    print("\nTo get nutritional data, you need a USDA API key (it's free):")
    print("1. Go to: https://fdc.nal.usda.gov/api-key-signup.html")
    print("2. Sign up and get your API key")
    print("3. Create a file: .streamlit/secrets.toml")
    print("4. Add this line: USDA_API_KEY = \"your_key_here\"")
    print("\nOR set environment variable:")
    print("   export USDA_API_KEY='your_key_here'  (Mac/Linux)")
    print("   set USDA_API_KEY=your_key_here       (Windows)")
    print("\nFor now, we'll continue with limited nutritional data...")
else:
    print(f"USDA API key available: {USDA_API_KEY[:1]}...")


# USDA API FUNCTIONS (from your finalapp.py)


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
        print(f"  ⚠ USDA API error for '{query}': {e}")
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
            "serving_size_g": 100.0,
            "name": data.get('description', '').strip().lower()
        }
        
        for nutrient in data.get('foodNutrients', []):
            nutrient_name = nutrient.get('nutrient', {}).get('name', '').lower()
            amount = nutrient.get('amount', 0)
            
            if 'phenylalanine' in nutrient_name:
                nutrients['phe_mg_per_100g'] = float(amount)
            elif nutrient_name == 'protein':
                nutrients['protein_g_per_100g'] = float(amount)
            elif nutrient_name == 'energy':
                # Convert kJ to kcal if needed
                unit = nutrient.get('nutrient', {}).get('unitName', '').lower()
                if 'kj' in unit:
                    nutrients['energy_kcal_per_100g'] = float(amount) * 0.239006
                else:
                    nutrients['energy_kcal_per_100g'] = float(amount)
        
        # If PHE not provided by USDA, calculate from protein (50 mg per gram of protein)
        if nutrients['phe_mg_per_100g'] == 0.0 and nutrients['protein_g_per_100g'] > 0:
            nutrients['phe_mg_per_100g'] = nutrients['protein_g_per_100g'] * 50.0
        
        return nutrients
    except Exception as e:
        print(f"  ⚠ Error fetching food details for ID {fdc_id}: {e}")
        return None

def search_usda_fallback(name):
    """Search USDA API for ingredient nutrition data (from your finalapp.py)"""
    
    # Clean up the search term
    search_term = name.lower().strip()
    
    # For common ingredients, use specific search terms
    search_improvements = {
        'mint': 'spearmint',
        'basil': 'basil fresh',
        'parsley': 'parsley fresh',
        'cilantro': 'coriander',
        'coriander': 'coriander',
        'olive oil': 'oil olive',
        'vegetable oil': 'oil vegetable',
        'canola oil': 'oil canola',
        'coconut oil': 'oil coconut',
        'salt': 'salt table',
        'pepper': 'pepper black',
        'black pepper': 'pepper black',
        'lemon': 'lemon raw without peel',
        'lime': 'lime raw',
        'garlic': 'garlic raw',
        'onion': 'onion raw',
        'tomato': 'tomato red ripe raw',
        'ginger': 'ginger root raw',
    }
    
    # Check if we have a better search term
    for key, improved in search_improvements.items():
        if key == search_term or key in search_term:
            search_term = improved
            break
    
    usda_results = search_usda_foods(search_term)
    if not usda_results:
        return None
    
    # Prioritize "Foundation" and "SR Legacy" foods (most complete data)
    exclude_keywords = ['candy', 'candies', 'nestle', 'kraft', 'kellogg', 'general mills', 
                       'chocolate', 'snack', 'cookie', 'cake', 'ice cream', 'dessert',
                       'egg', 'chicken', 'beef', 'pork', 'fish', 'cheese', 'milk', 'yogurt',
                       'supplement', 'formula', 'beverage', 'drink', 'soda']
    
    original_term = name.lower().strip().split()[0]
    
    priority_results = []
    acceptable_results = []
    
    for result in usda_results[:10]:
        description = result.get('description', '').lower()
        data_type = result.get('dataType', '')
        
        # For herbs/spices, ensure the original term is in the description
        if original_term in ['mint', 'basil', 'parsley', 'cilantro', 'coriander', 'thyme', 
                            'rosemary', 'oregano', 'sage', 'dill']:
            if original_term not in description and 'spearmint' not in description:
                continue
        
        # Skip unwanted categories
        if any(keyword in description for keyword in exclude_keywords):
            continue
            
        if data_type in ['Foundation', 'SR Legacy']:
            priority_results.append(result)
        elif data_type in ['Survey (FNDDS)']:
            acceptable_results.append(result)
    
    # Try priority results first, then acceptable ones
    for result in (priority_results + acceptable_results):
        fdc_id = result.get('fdcId')
        if fdc_id:
            details = get_usda_food_details(fdc_id)
            if details is not None:
                details['usda_match'] = result.get('description', 'Unknown')
                details['usda_data_type'] = result.get('dataType', 'Unknown')
                return details
    
    return None


# STEP 0.1: LOAD AND ANALYZE RATINGS DATA


print("STEP 0.1: LOADING AND ANALYZING RATINGS DATA")

ratings_file = "ratingsappdata - Sheet1.csv"

if not os.path.exists(ratings_file):
    print(f"\nERROR: File '{ratings_file}' not found!")
    print("Make sure the CSV file is in the same directory as this script.")
    exit()

ratings_df = pd.read_csv(ratings_file)

print(f"\nLoaded {ratings_file}")
print(f"  - Total rows: {len(ratings_df)}")
print(f"  - Total columns: {len(ratings_df.columns)}")
print(f"  - Total users: {ratings_df['Name'].nunique()}")


# EXTRACT FOOD RATINGS



print("Extracting food ratings from wide format...")


demographic_cols = [
    'Name', 'Email', 'Age', 'Gender', 'Height_cm', 'Weight_kg', 
    'PHE_tolerance', 'Dietary_tolerance_mg_per_day',
    'Selected_Cuisines', 'Timestamp'
]

food_columns = []
for col in ratings_df.columns:
    if col not in demographic_cols and '(' in col and ')' in col:
        food_columns.append(col)

print(f"\nFound {len(food_columns)} food rating columns")

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

print(f"\nExtracted ratings to long format")
print(f"  - Total user-food rating pairs: {len(user_food_df)}")
print(f"  - Unique users: {user_food_df['user_name'].nunique()}")
print(f"  - Unique foods: {user_food_df['food'].nunique()}")

print(f"\nRating distribution:")
rating_counts = user_food_df['rating'].value_counts().sort_index()
for rating, count in rating_counts.items():
    percentage = (count / len(user_food_df)) * 100
    print(f"  {rating}: {count:4d} ({percentage:5.1f}%)")

ratings_per_user = user_food_df.groupby('user_name').size()
print(f"\nRatings per user:")
print(f"  Mean:   {ratings_per_user.mean():.1f}")
print(f"  Median: {ratings_per_user.median():.1f}")
print(f"  Min:    {ratings_per_user.min()}")
print(f"  Max:    {ratings_per_user.max()}")

ratings_per_food = user_food_df.groupby('food').size()
print(f"\nRatings per food:")
print(f"  Mean:   {ratings_per_food.mean():.1f}")
print(f"  Median: {ratings_per_food.median():.1f}")
print(f"  Foods rated by only 1 user: {(ratings_per_food == 1).sum()}")
print(f"  Foods rated by 3+ users: {(ratings_per_food >= 3).sum()}")

user_food_df.to_csv('data_user_food_ratings.csv', index=False)
print(f"\nSaved to: data_user_food_ratings.csv")


# STEP 0.2: EXTRACT USER PROFILES AND CALCULATE LIMITS


print("STEP 0.2: EXTRACTING USER PROFILES AND CALCULATING LIMITS")

user_profiles = []

for idx, row in ratings_df.iterrows():
    profile = {
        'user_name': row['Name'],
        'user_email': row['Email'],
        'age': row.get('Age', None),
        'gender': row.get('Gender', None),
        'height_cm': row.get('Height_cm', None),
        'weight_kg': row.get('Weight_kg', None),
        'phe_tolerance_mg_per_day': row.get('Dietary_tolerance_mg_per_day', 
                                            row.get('PHE_tolerance', None)),
        'selected_cuisines': row.get('Selected_Cuisines', '')
    }
    user_profiles.append(profile)

user_profiles_df = pd.DataFrame(user_profiles)

print(f"\nExtracted profiles for {len(user_profiles_df)} users")

missing_counts = user_profiles_df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"\nMissing values in user profiles:")
    for col, count in missing_counts.items():
        if count > 0:
            percentage = (count / len(user_profiles_df)) * 100
            print(f"  {col}: {count} ({percentage:.1f}%)")


# CALCULATE NUTRITIONAL LIMITS



print("Calculating personalized nutritional limits...")


def get_infant_daily_needs(age_months, weight_kg):
    needs = {}
    if age_months < 3:
        needs.update({
            'protein_g_per_kg': 3.5, 
            'phe_mg_per_kg_min': 25, 
            'phe_mg_per_kg_max': 70, 
            'energy_kcal_per_kg': 120,
            'age_group': '0-3 months'
        })
    elif age_months < 6:
        needs.update({
            'protein_g_per_kg': 3.5,
            'phe_mg_per_kg_min': 20,
            'phe_mg_per_kg_max': 45,
            'energy_kcal_per_kg': 120,
            'age_group': '3-6 months'
        })
    elif age_months < 9:
        needs.update({
            'protein_g_per_kg': 3.0,
            'phe_mg_per_kg_min': 15,
            'phe_mg_per_kg_max': 35,
            'energy_kcal_per_kg': 110,
            'age_group': '6-9 months'
        })
    else:
        needs.update({
            'protein_g_per_kg': 3.0,
            'phe_mg_per_kg_min': 10,
            'phe_mg_per_kg_max': 35,
            'energy_kcal_per_kg': 105,
            'age_group': '9-12 months'
        })
    
    needs['protein_g'] = needs['protein_g_per_kg'] * weight_kg
    needs['phe_mg_min'] = needs['phe_mg_per_kg_min'] * weight_kg
    needs['phe_mg_max'] = needs['phe_mg_per_kg_max'] * weight_kg
    needs['energy_kcal'] = needs['energy_kcal_per_kg'] * weight_kg
    
    return needs

def get_child_adult_daily_needs(age_months, weight_kg, sex):
    needs = {}
    if age_months < 48:
        needs.update({
            'phe_mg_min': 200, 
            'phe_mg_max': 400, 
            'protein_g': 30, 
            'energy_kcal': 1300,
            'age_group': '1-4 years'
        })
    elif age_months < 84:
        needs.update({
            'phe_mg_min': 210,
            'phe_mg_max': 450,
            'protein_g': 35,
            'energy_kcal': 1700,
            'age_group': '4-7 years'
        })
    elif age_months < 132:
        needs.update({
            'phe_mg_min': 220,
            'phe_mg_max': 500,
            'protein_g': 40,
            'energy_kcal': 2400,
            'age_group': '7-11 years'
        })
    elif age_months < 180:
        if sex == "Female":
            needs.update({
                'phe_mg_min': 250,
                'phe_mg_max': 750,
                'protein_g': 50,
                'energy_kcal': 2200,
                'age_group': '11-15 years'
            })
        else:
            needs.update({
                'phe_mg_min': 225,
                'phe_mg_max': 900,
                'protein_g': 55,
                'energy_kcal': 2700,
                'age_group': '11-15 years'
            })
    elif age_months < 228:
        if sex == "Female":
            needs.update({
                'phe_mg_min': 230,
                'phe_mg_max': 700,
                'protein_g': 55,
                'energy_kcal': 2100,
                'age_group': '15-19 years'
            })
        else:
            needs.update({
                'phe_mg_min': 295,
                'phe_mg_max': 1100,
                'protein_g': 65,
                'energy_kcal': 2800,
                'age_group': '15-19 years'
            })
    else:
        if sex == "Female":
            needs.update({
                'phe_mg_min': 220,
                'phe_mg_max': 700,
                'protein_g': 60,
                'energy_kcal': 2100,
                'age_group': '19+ years'
            })
        else:
            needs.update({
                'phe_mg_min': 290,
                'phe_mg_max': 1200,
                'protein_g': 70,
                'energy_kcal': 2900,
                'age_group': '19+ years'
            })
    
    return needs

def calculate_user_nutritional_limits(user_profile):
    age = user_profile['age']
    weight_kg = user_profile['weight_kg']
    gender = user_profile['gender']
    
    if pd.isna(age) or pd.isna(weight_kg):
        return None
    
    age_months = age * 12
    
    if age_months < 12:
        return get_infant_daily_needs(age_months, weight_kg)
    else:
        return get_child_adult_daily_needs(age_months, weight_kg, gender)

user_limits_list = []

for idx, user_profile in user_profiles_df.iterrows():
    limits = calculate_user_nutritional_limits(user_profile)
    
    if limits is not None:
        limits['user_name'] = user_profile['user_name']
        limits['user_email'] = user_profile['user_email']
        limits['age'] = user_profile['age']
        limits['weight_kg'] = user_profile['weight_kg']
        limits['gender'] = user_profile['gender']
        user_limits_list.append(limits)

user_limits_df = pd.DataFrame(user_limits_list)

print(f"\nCalculated nutritional limits for {len(user_limits_df)} users")

print(f"\nSample of calculated limits:")
sample_cols = ['user_name', 'age_group', 'phe_mg_min', 'phe_mg_max', 'protein_g', 'energy_kcal']
print(user_limits_df[sample_cols].head(10).to_string(index=False))

user_limits_df.to_csv('data_user_nutritional_limits.csv', index=False)
print(f"\nSaved to: data_user_nutritional_limits.csv")


# STEP 0.3: BUILD FOOD DATABASE USING USDA API


print("STEP 0.3: BUILDING FOOD DATABASE USING USDA API")


unique_foods = user_food_df['food'].unique()
print(f"\nTotal unique foods to fetch: {len(unique_foods)}")

if USDA_API_KEY is None:
    print("\n⚠ WARNING: No USDA API key available!")
    print("  Cannot fetch nutritional data.")
    print("  Creating empty database...")
    food_db = {}
else:
    print(f"\nFetching nutritional data from USDA...")
    print("  This may take a few minutes...")
    
    food_db = {}
    foods_found = 0
    foods_missing = []
    
    for i, food in enumerate(unique_foods, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(unique_foods)} foods processed...")
        
        # Clean food name (remove cuisine tag)
        clean_name = food.lower().strip()
        if '(' in clean_name:
            clean_name = clean_name.split('(')[0].strip()
        
        # Search USDA
        nutrients = search_usda_fallback(clean_name)
        
        if nutrients:
            food_db[clean_name] = nutrients
            foods_found += 1
        else:
            foods_missing.append(food)
    
    print(f"\nUSDA data collection complete")
    print(f"  - Foods found: {foods_found} ({foods_found/len(unique_foods)*100:.1f}%)")
    print(f"  - Foods missing: {len(foods_missing)} ({len(foods_missing)/len(unique_foods)*100:.1f}%)")

# Save food database
if food_db:
    food_db_df = pd.DataFrame([
        {'food_name': name, **nutrients}
        for name, nutrients in food_db.items()
    ])
    food_db_df.to_csv('data_food_database.csv', index=False)
    print(f"\nSaved to: data_food_database.csv")

# Save missing foods
if 'foods_missing' in locals() and foods_missing:
    missing_df = pd.DataFrame({'food': foods_missing})
    missing_df.to_csv('data_missing_foods.csv', index=False)
    print(f"Missing foods list saved to: data_missing_foods.csv")


# FINAL SUMMARY


print("STAGE 0 COMPLETE - SUMMARY")


coverage_pct = (len(food_db) / len(unique_foods)) * 100 if len(unique_foods) > 0 else 0

print(f"""
Data Files Created:
  1. data_user_food_ratings.csv       - {len(user_food_df)} ratings
  2. data_user_nutritional_limits.csv - {len(user_limits_df)} user profiles with limits
  3. data_food_database.csv           - {len(food_db)} foods with nutritional data
  4. data_missing_foods.csv           - {len(foods_missing) if 'foods_missing' in locals() else 0} foods needing data

Statistics:
  • Total users: {user_food_df['user_name'].nunique()}
  • Total ratings: {len(user_food_df)}
  • Users with complete profiles: {len(user_limits_df)}
  • Food coverage: {coverage_pct:.1f}%

Next Steps:
""")

if coverage_pct >= 80:
    print("  Good food coverage! Ready for Stage 1: Create train/test splits")
elif USDA_API_KEY:
    print("  ⚠ Food coverage is moderate. You can proceed, but results may be limited.")
    print("  Consider manually adding data for missing foods.")
else:
    print("  No USDA API key - cannot fetch nutritional data!")
    print("  Please add API key and re-run Stage 0.")

print("="*80)