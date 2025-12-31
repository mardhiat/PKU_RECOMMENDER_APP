import streamlit as st
import pandas as pd
from datetime import datetime, date
import os
import requests

# -------------------------------------------------------
# App config
# -------------------------------------------------------
st.set_page_config(page_title="PKU Diet Manager", layout="wide")

# -------------------------------------------------------
# USDA API Integration
# -------------------------------------------------------

# 🔑 Get API key from secrets
try:
    USDA_API_KEY = st.secrets["USDA_API_KEY"]
except (KeyError, FileNotFoundError):
    USDA_API_KEY = None
    st.warning("⚠️ USDA API key not found in secrets.toml. Add it to enable food database search.")

@st.cache_data(ttl=86400)  # Cache for 24 hours
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
        st.warning(f"USDA API error: {e}")
        return []

@st.cache_data(ttl=86400)
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
                # Convert kJ to kcal if needed
                unit = nutrient.get('nutrient', {}).get('unitName', '').lower()
                if 'kj' in unit:
                    nutrients['energy(kcal)'] = float(amount) * 0.239006
                else:
                    nutrients['energy(kcal)'] = float(amount)
        
        # If PHE not provided by USDA, calculate from protein
        # PHE makes up approximately 5% of total protein (50 mg per gram of protein)
        if nutrients['phe(mg)'] == 0.0 and nutrients['protein(g)'] > 0:
            nutrients['phe(mg)'] = nutrients['protein(g)'] * 50.0
            nutrients['phe_calculated'] = True  # Flag that this was calculated
        else:
            nutrients['phe_calculated'] = False
        
        return pd.Series(nutrients)
    except Exception as e:
        st.warning(f"Error fetching food details: {e}")
        return None

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

@st.cache_data
def load_consolidated_foods():
    """Load consolidated nutrient database - OPTIONAL fallback only"""
    try:
        df = pd.read_csv("consolidated_chat_ingredients.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        
        if "ingredient" in df.columns:
            df = df.rename(columns={"ingredient": "name"})
            
        for c in ["phe(mg)", "protein(g)", "energy(kcal)", "serving_size(g)"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                
        if "name" in df.columns:
            df["name"] = df["name"].astype(str).str.strip().str.lower()

        return df
    except Exception as e:
        # If CSV doesn't exist or is bad, return empty DataFrame - we'll use USDA instead
        st.info("💡 Using USDA database for all ingredient nutrition data")
        return pd.DataFrame()

@st.cache_data
def load_cuisine_files():
    """Load all cuisine CSV files from current directory."""
    cuisine_data = {}
    cuisine_files = {
        "African": "african_foods.csv",
        "Central European": "central_european_foods.csv",
        "Chinese": "chinese_foods.csv",
        "Eastern European": "eastern_european_foods.csv",
        "Indian": "indian_foods.csv",
        "Italian": "italian_foods.csv",
        "Japanese": "japanese_foods.csv",
        "Mediterranean": "mediterranean_foods.csv",
        "Mexican": "mexican_foods.csv",
        "Scottish": "scottish_foods.csv",
    }

    for cuisine_name, filename in cuisine_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df.columns = [c.strip().lower() for c in df.columns]

                rename_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if cl == "meal": rename_map[c] = "dish"
                    elif cl == "ingredient": rename_map[c] = "ingredient"
                    elif cl in ["grams", "weight (g)", "weight", "amount"]: rename_map[c] = "amount"
                    elif cl in ["serving size (g)", "serving size"]: rename_map[c] = "Serving Size (g)"
                    elif cl in ["number of ingredient", "ingredients #"]: rename_map[c] = "ingredients_n"
                    elif cl in ["meal type", "Meal Type"]: rename_map[c] = "Meal Type"
                if rename_map:
                    df = df.rename(columns=rename_map)

                if "dish" in df.columns:
                    df["dish"] = df["dish"].ffill()

                if "Meal Type" not in df.columns:
                    df["Meal Type"] = "ALL"
                    
                for needed in ["dish", "ingredient", "amount", "Meal Type"]:
                    if needed not in df.columns:
                        df[needed] = None
                cuisine_data[cuisine_name] = df
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    return cuisine_data

consolidated_db = load_consolidated_foods()
cuisine_db = load_cuisine_files()

# -------------------------------------------------------
# Static baby foods
# -------------------------------------------------------

BABY_FOODS = {
    "Vegetables": {
        "Carrots, cooked": {"weight_g": 39, "phe_mg": 14, "protein_g": 0.4, "calories": 18},
        "Cauliflower, cooked": {"weight_g": 23, "phe_mg": 16, "protein_g": 0.4, "calories": 6},
        "Green beans, cooked": {"weight_g": 23, "phe_mg": 15, "protein_g": 0.4, "calories": 9},
        "Squash, cooked": {"weight_g": 50, "phe_mg": 15, "protein_g": 0.4, "calories": 17},
    },
    "Fruits": {
        "Applesauce, sweetened": {"weight_g": 86, "phe_mg": 5, "protein_g": 0.2, "calories": 48},
        "Bananas": {"weight_g": 47, "phe_mg": 15, "protein_g": 0.5, "calories": 47},
        "Peaches": {"weight_g": 88, "phe_mg": 15, "protein_g": 0.6, "calories": 38},
        "Pears": {"weight_g": 68, "phe_mg": 15, "protein_g": 0.3, "calories": 39},
    },
}

TABLE_FOODS = {
    "Vegetables": {
        "Broccoli, cooked (2 Tbsp)": {"weight_g": 20, "phe_mg": 18, "protein_g": 0.6, "calories": 6},
        "Butternut squash, mashed (2 Tbsp)": {"weight_g": 30, "phe_mg": 15, "protein_g": 0.4, "calories": 12},
        "Zucchini (summer squash), cooked (1/4 cup)": {"weight_g": 45, "phe_mg": 15, "protein_g": 0.4, "calories": 9},
        "Carrots, fresh or cooked (1/4 cup)": {"weight_g": 39, "phe_mg": 14, "protein_g": 0.4, "calories": 18},
    },
    "Fruits": {
        "Banana, sliced (3 Tbsp)": {"weight_g": 42, "phe_mg": 16, "protein_g": 0.4, "calories": 39},
        "Watermelon, cubed (3/4 cup)": {"weight_g": 120, "phe_mg": 18, "protein_g": 0.7, "calories": 38},
        "Applesauce, sweetened (1/4 cup + 2 Tbsp)": {"weight_g": 86, "phe_mg": 5, "protein_g": 0.2, "calories": 48},
    },
    "Breads/Cereals": {
        "Rice (prepared), 2 Tbsp": {"weight_g": 25, "phe_mg": 32, "protein_g": 0.6, "calories": 27},
        "Macaroni (cooked), 1 Tbsp + 1.5 tsp": {"weight_g": 12, "phe_mg": 31, "protein_g": 0.6, "calories": 18},
        "Corn Flakes (1/3 cup)": {"weight_g": 7, "phe_mg": 31, "protein_g": 0.6, "calories": 29},
    },
}

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def calculate_age_months(y, m, d):
    today = date.today()
    b = date(int(y), int(m), int(d))
    months = (today.year - b.year) * 12 + (today.month - b.month)
    if today.day < b.day:
        months -= 1
    return max(0, months)

def format_age_display(birth_date):
    today = date.today()
    days = (today - birth_date).days
    months = days // 30
    years = months // 12
    if days < 30:
        return f"{days} days"
    elif months < 12:
        return f"{months} months"
    else:
        return f"{years} years {months % 12} months"

def get_infant_daily_needs(age_months, weight_kg):
    needs = {}
    if age_months < 3:
        needs.update({'protein_g_per_kg': 3.5, 'phe_mg_per_kg_min': 25, 'phe_mg_per_kg_max': 70, 'energy_kcal_per_kg': 120, 'fluid_ml_per_kg': 160, 'age_group': '0-3 months'})
    elif age_months < 6:
        needs.update({'protein_g_per_kg': 3.5, 'phe_mg_per_kg_min': 20, 'phe_mg_per_kg_max': 45, 'energy_kcal_per_kg': 120, 'fluid_ml_per_kg': 160, 'age_group': '3-6 months'})
    elif age_months < 9:
        needs.update({'protein_g_per_kg': 3.0, 'phe_mg_per_kg_min': 15, 'phe_mg_per_kg_max': 35, 'energy_kcal_per_kg': 110, 'fluid_ml_per_kg': 145, 'age_group': '6-9 months'})
    else:
        needs.update({'protein_g_per_kg': 3.0, 'phe_mg_per_kg_min': 10, 'phe_mg_per_kg_max': 35, 'energy_kcal_per_kg': 105, 'fluid_ml_per_kg': 135, 'age_group': '9-12 months'})
    needs['protein_g'] = needs['protein_g_per_kg'] * weight_kg
    needs['phe_mg_min'] = needs['phe_mg_per_kg_min'] * weight_kg
    needs['phe_mg_max'] = needs['phe_mg_per_kg_max'] * weight_kg
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    needs['energy_kcal'] = needs['energy_kcal_per_kg'] * weight_kg
    needs['fluid_ml'] = needs['fluid_ml_per_kg'] * weight_kg
    return needs

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
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    return needs

def calculate_milk_amount(phe_target_mg, milk_type, split_ratio=0.5):
    if milk_type is None:
        milk_type = "Breast Milk (Human Milk)"
    
    if milk_type == "Breast Milk (Human Milk)":
        phe_per_100ml, protein_per_100ml, energy_per_100ml = 48, 1.07, 72
        ml = (phe_target_mg / phe_per_100ml) * 100.0
        return {"milk_type": milk_type, "milk_ml": ml,
                "phe_mg": phe_target_mg,
                "protein_g": (ml/100.0)*protein_per_100ml,
                "calories_kcal": (ml/100.0)*energy_per_100ml}
    elif milk_type == "Similac With Iron":
        phe_per_100ml, protein_per_100ml, energy_per_100ml = 59, 1.40, 68
        ml = (phe_target_mg / phe_per_100ml) * 100.0
        return {"milk_type": milk_type, "milk_ml": ml,
                "phe_mg": phe_target_mg,
                "protein_g": (ml/100.0)*protein_per_100ml,
                "calories_kcal": (ml/100.0)*energy_per_100ml}
    elif "Both" in str(milk_type):
        phe_breast = phe_target_mg * split_ratio
        phe_similac = phe_target_mg * (1 - split_ratio)
        breast = calculate_milk_amount(phe_breast, "Breast Milk (Human Milk)")
        similac = calculate_milk_amount(phe_similac, "Similac With Iron")
        return {"milk_type": milk_type,
                "milk_ml": breast["milk_ml"] + similac["milk_ml"],
                "phe_mg": breast["phe_mg"] + similac["phe_mg"],
                "protein_g": breast["protein_g"] + similac["protein_g"],
                "calories_kcal": breast["calories_kcal"] + similac["calories_kcal"]}
    else:
        return {"milk_type": "Breast Milk (Human Milk)", "milk_ml": 0,
                "phe_mg": 0, "protein_g": 0, "calories_kcal": 0}

def compute_medical_food_gap(protein_needed_g, protein_from_food_g, age_months):
    protein_gap = max(0.0, protein_needed_g - protein_from_food_g)
    if age_months < 24:
        kcal_per_g_powder = 4.8
        protein_per_100g = 15.0
    else:
        kcal_per_g_powder = 4.1
        protein_per_100g = 30.0
    grams_powder = (protein_gap * 100.0) / protein_per_100g if protein_per_100g > 0 else 0.0
    calories_kcal = grams_powder * kcal_per_g_powder
    return {
        "protein_gap_g": protein_gap,
        "estimated_powder_g": grams_powder,
        "estimated_calories_kcal": calories_kcal
    }

# -------------------------------------------------------
# Ingredient mapping with USDA
# -------------------------------------------------------

def normalize_name(s):
    return str(s).strip().lower()

def parse_portion(text):
    if text is None or text == "" or str(text).lower() == "nan":
        return None
    
    if isinstance(text, (int, float)):
        return float(text) if text > 0 else None
    
    s = str(text).strip().lower()
    if not s or s == "nan":
        return None
    
    tokens = s.split()
    try:
        qty = float(tokens[0])
        return qty if qty > 0 else None
    except (ValueError, IndexError):
        return None

def search_usda_fallback(name):
    """Search USDA API for ingredient nutrition data with improved matching"""
    
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
    # Avoid branded foods and unwanted categories
    exclude_keywords = ['candy', 'candies', 'nestle', 'kraft', 'kellogg', 'general mills', 
                       'chocolate', 'snack', 'cookie', 'cake', 'ice cream', 'dessert',
                       'egg', 'chicken', 'beef', 'pork', 'fish', 'cheese', 'milk', 'yogurt',
                       'supplement', 'formula', 'beverage', 'drink', 'soda']
    
    # For herbs, require the original term to be in the description
    original_term = name.lower().strip().split()[0]  # First word of original search
    
    priority_results = []
    acceptable_results = []
    
    for result in usda_results[:10]:  # Check first 10 results
        description = result.get('description', '').lower()
        data_type = result.get('dataType', '')
        
        # For herbs/spices, ensure the original term is actually in the description
        if original_term in ['mint', 'basil', 'parsley', 'cilantro', 'coriander', 'thyme', 
                            'rosemary', 'oregano', 'sage', 'dill']:
            # Check if the herb name is in the description
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
                # Store which USDA food was matched for debugging
                details['usda_match'] = result.get('description', 'Unknown')
                details['usda_data_type'] = result.get('dataType', 'Unknown')
                return details
    
    return None

def select_best_match(name, food_db, use_usda=True):
    """Find best matching ingredient - prioritize USDA API over local DB"""
    
    # First try USDA API (most accurate and up-to-date)
    if use_usda and USDA_API_KEY is not None:
        usda_result = search_usda_fallback(name)
        if usda_result is not None:
            return usda_result
    
    # Fallback to local CSV if USDA fails or API unavailable
    if food_db.empty or "name" not in food_db.columns:
        return None
    
    target = normalize_name(name)
    names_col = food_db["name"]
    
    def prefer_nonzero(matches):
        if matches.empty:
            return None
        non_zero = matches[(matches["phe(mg)"] > 0) | (matches["protein(g)"] > 0) | (matches["energy(kcal)"] > 0)]
        return non_zero.iloc[0] if not non_zero.empty else matches.iloc[0]
    
    # Try exact match
    exact = food_db[names_col == target]
    if len(exact) >= 1:
        return prefer_nonzero(exact)
    
    # Try startswith
    sw = food_db[names_col.str.startswith(target, na=False)]
    if len(sw) >= 1:
        return prefer_nonzero(sw)
    
    # Try contains
    ct = food_db[names_col.str.contains(target, na=False, regex=False)]
    if len(ct) >= 1:
        return prefer_nonzero(ct)
    
    # Try reverse contains
    for idx, db_name in enumerate(names_col):
        if db_name and len(db_name) > 3 and db_name in target:
            match_row = food_db.iloc[idx]
            if match_row["phe(mg)"] > 0 or match_row["protein(g)"] > 0 or match_row["energy(kcal)"] > 0:
                return match_row
    
    return None

def scale_nutrients(row, weight_g):
    serving_size = row.get("serving_size(g)", 100.0)
    if serving_size == 0:
        serving_size = 100.0
    
    phe_serv = row.get("phe(mg)", 0.0)
    prot_serv = row.get("protein(g)", 0.0)
    cal_serv = row.get("energy(kcal)", 0.0)

    phe_per_g = (phe_serv / serving_size)
    prot_per_g = (prot_serv / serving_size)
    cal_per_g = (cal_serv / serving_size)
    
    return {
        "weight_g": weight_g,
        "phe_mg": phe_per_g * weight_g,
        "protein_g": prot_per_g * weight_g,
        "calories": cal_per_g * weight_g,
    }

def compute_dish_nutrients(dish_df, food_db):
    total_phe = 0.0
    total_protein = 0.0
    total_calories = 0.0
    total_weight = 0.0
    ingredients_list = []
    
    for _, row in dish_df.iterrows():
        ing_name = str(row.get("ingredient", "")).strip()
        amount = parse_portion(str(row.get("amount", "")))
        if not ing_name or amount is None:
            continue
            
        match = select_best_match(ing_name, food_db, use_usda=True)
        
        if match is None:
            ingredients_list.append({
                "name": ing_name,
                "weight_g": amount,
                "phe_mg": 0.0,
                "protein_g": 0.0,
                "calories": 0.0,
                "note": "Not found in USDA database",
                "usda_match": None
            })
            total_weight += amount
            continue
            
        scaled = scale_nutrients(match, amount)
        ingredient_entry = {
            "name": match["name"],
            "usda_match": match.get("usda_match", None),
            "usda_data_type": match.get("usda_data_type", None),
            **scaled
        }
        ingredients_list.append(ingredient_entry)
        total_phe += scaled["phe_mg"]
        total_protein += scaled["protein_g"]
        total_calories += scaled["calories"]
        total_weight += scaled["weight_g"]
        
    return {
        "ingredients": ingredients_list,
        "totals": {
            "phe_mg": total_phe,
            "protein_g": total_protein,
            "calories": total_calories,
            "weight_g": total_weight
        }
    }

def update_daily_totals():
    totals = {"phe_mg": 0.0, "protein_g": 0.0, "calories": 0.0}
    for food in st.session_state.selected_foods_list:
        totals["phe_mg"] += food.get("phe_mg", 0.0)
        totals["protein_g"] += food.get("protein_g", 0.0)
        totals["calories"] += food.get("calories", 0.0)
    st.session_state.daily_totals = totals

def calculate_baby_diet_with_solids(age_months, weight_kg, milk_type, solid_foods):
    needs = get_infant_daily_needs(age_months, weight_kg)
    total_solid_phe = sum(food['phe_mg'] for food in solid_foods)
    total_solid_protein = sum(food['protein_g'] for food in solid_foods)
    total_solid_calories = sum(food['calories'] for food in solid_foods)
    
    phe_target = needs['phe_mg_max']
    remaining_phe = phe_target - total_solid_phe
    if remaining_phe < 0:
        remaining_phe = max(needs['phe_mg_min'] - total_solid_phe, 0)
        
    milk = calculate_milk_amount(remaining_phe,
                             st.session_state.user_milk_type,
                             st.session_state.get("milk_split_ratio", 0.5))
    total_phe = total_solid_phe + milk['phe_mg']
    
    if total_phe > needs['phe_mg_max']:
        safe_remaining = max(needs['phe_mg_max'] - total_solid_phe, 0)
        milk = calculate_milk_amount(safe_remaining, milk_type)
        
    total_protein_food_milk = total_solid_protein + milk['protein_g']
    medical_gap = compute_medical_food_gap(needs['protein_g'], total_protein_food_milk, age_months)
    
    result = {
        'needs': needs,
        'solid_foods': {
            'foods': solid_foods,
            'total_phe_mg': total_solid_phe,
            'total_protein_g': total_solid_protein,
            'total_calories': total_solid_calories
        },
        'milk': milk,
        'medical_food_gap': medical_gap,
        'totals': {
            'protein_g': total_protein_food_milk + medical_gap['protein_gap_g'],
            'phe_mg': total_solid_phe + milk['phe_mg'],
            'calories_kcal': total_solid_calories + milk['calories_kcal'] + medical_gap['estimated_calories_kcal']
        }
    }
    return result

def display_baby_diet_plan(result):
    st.markdown("---")
    st.subheader("📈 Daily totals")
    totals = result['totals']
    needs = result['needs']
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total protein", f"{totals['protein_g']:.2f} g", f"Target: {needs['protein_g']:.2f} g")
    with c2:
        in_range = needs['phe_mg_min'] <= totals['phe_mg'] <= needs['phe_mg_max']
        st.metric("Total PHE", f"{totals['phe_mg']:.0f} mg",
                  f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg" + (" ✅" if in_range else " ⚠️"))
    with c3:
        st.metric("Total calories", f"{totals['calories_kcal']:.0f} kcal", f"Target: {needs['energy_kcal']:.0f} kcal")

def add_custom_dish_ui(consolidated_db):
    st.markdown("#### Add a custom dish")
    with st.form("custom_dish_form", clear_on_submit=False):
        dish_name = st.text_input("Dish name")
        meal_type_label = st.selectbox("Assign to meal", ["Breakfast", "Lunch", "Dinner", "Snack", "ALL"], index=4)
        st.markdown("Enter ingredients (name and grams).")
        
        ing_cols = st.columns([3, 1])
        ingredients = []
        for i in range(8):
            with ing_cols[0]:
                n = st.text_input(f"Ingredient {i+1} name", key=f"c_ing_{i}")
            with ing_cols[1]:
                g = st.number_input(f"g {i+1}", min_value=0.0, step=1.0, value=0.0, key=f"c_g_{i}")
            if n and g > 0:
                ingredients.append({"ingredient": n, "amount": g})
                
        submitted = st.form_submit_button("➕ Add custom dish")
        
    if submitted and dish_name and ingredients:
        rows = []
        for ing in ingredients:
            row = {"dish": dish_name, "ingredient": ing["ingredient"], "amount": ing["amount"], "Meal Type": "ALL"}
            rows.append(row)
        dish_df = pd.DataFrame(rows)
        
        dish_nutrients = compute_dish_nutrients(dish_df, consolidated_db)
        totals = dish_nutrients["totals"]
        
        st.success(f"✅ Added custom dish '{dish_name}'")
        return {
            "meal": meal_type_label,
            "name": dish_name,
            "weight_g": totals["weight_g"],
            "phe_mg": totals["phe_mg"],
            "protein_g": totals["protein_g"],
            "calories": totals["calories"],
            "ingredients": dish_nutrients["ingredients"]
        }
    return None

# -------------------------------------------------------
# Main app
# -------------------------------------------------------

def main():
    if 'profile_created' not in st.session_state:
        st.session_state.profile_created = False
    if 'solid_foods_list' not in st.session_state:
        st.session_state.solid_foods_list = []
    if 'selected_foods_list' not in st.session_state:
        st.session_state.selected_foods_list = []

    # Check API key
    if USDA_API_KEY is None:
        st.info("💡 To enable USDA food database search, add your API key to `.streamlit/secrets.toml`. Get one free at: https://fdc.nal.usda.gov/api-key-signup.html")

    if not st.session_state.profile_created:
        st.title("PKU Diet Management System")
        st.markdown("Plan safe PKU diets with USDA nutrition database and medical food calculations.")
        st.info("🌍 Nutrition data from USDA FoodData Central")
        st.markdown("---")
        st.header("Create profile")

        age_category = st.radio("Profile type:", ["Baby (0-12 months)", "Child (1-12 years)", "Adult (12+ years)"])
        sex = st.radio("Sex:", ["Male", "Female"]) if age_category != "Baby (0-12 months)" else "Male"

        col1, col2 = st.columns(2)
        with col1:
            units = st.radio("Units:", ["Metric", "Imperial"])
            if units == "Metric":
                weight = st.number_input('Weight (kg):', min_value=0.0, step=0.1,
                                         value=7.0 if age_category == "Baby (0-12 months)" else 20.0)
                height_cm = st.number_input('Height (cm):', min_value=0.0, step=1.0,
                                            value=65.0 if age_category == "Baby (0-12 months)" else 120.0)
            else:
                weight_lbs = st.number_input('Weight (lbs):', min_value=0.0, step=0.1,
                                             value=15.4 if age_category == "Baby (0-12 months)" else 44.0)
                height_in = st.number_input('Height (in):', min_value=0.0, step=0.5,
                                            value=25.6 if age_category == "Baby (0-12 months)" else 47.0)
                weight = weight_lbs * 0.453592
                height_cm = height_in * 2.54
        with col2:
            birth_year = st.number_input('Birth year:', min_value=1900, max_value=datetime.now().year,
                                         value=2024 if age_category == "Baby (0-12 months)" else 2017)
            birth_month = st.number_input('Birth month:', min_value=1, max_value=12,
                                          value=6 if age_category == "Baby (0-12 months)" else 1)
            birth_day = st.number_input('Birth day:', min_value=1, max_value=31, value=1)
            current_phe = st.number_input('Current blood PHE (mg/dL):', min_value=0.0, step=0.1, value=5.0)

        milk_type = None
        if age_category == "Baby (0-12 months)":
            milk_type = st.radio("Milk type:", ["Breast Milk (Human Milk)","Similac With Iron","Both"])
            
            if milk_type == "Both":
                split_ratio = st.slider("Proportion breast milk vs. Similac", 0.0, 1.0, 0.5)
                st.session_state.milk_split_ratio = split_ratio
                milk_type = f"Both (Breast {split_ratio*100:.0f}%, Similac {(1-split_ratio)*100:.0f}%)"

        if st.button("Create profile and start planning", type="primary"):
            if weight <= 0 or height_cm <= 0 or current_phe <= 0:
                st.error("Please enter valid weight, height, and current PHE.")
            else:
                st.session_state.profile_created = True
                st.session_state.user_age_category = age_category
                st.session_state.user_sex = sex
                st.session_state.user_weight = weight
                st.session_state.user_height_cm = height_cm
                st.session_state.user_birth_year = birth_year
                st.session_state.user_birth_month = birth_month
                st.session_state.user_birth_day = birth_day
                st.session_state.user_current_phe = current_phe
                st.session_state.user_milk_type = milk_type
                st.rerun()
    else:
        # Sidebar profile
        st.sidebar.header("👤 Profile")
        bdate = date(int(st.session_state.user_birth_year), int(st.session_state.user_birth_month), int(st.session_state.user_birth_day))
        st.sidebar.write(f"**Age:** {format_age_display(bdate)}")
        st.sidebar.write(f"**Weight:** {st.session_state.user_weight:.1f} kg")
        st.sidebar.write(f"**Height:** {st.session_state.user_height_cm:.1f} cm")
        st.sidebar.write(f"**Current PHE:** {st.session_state.user_current_phe:.1f} mg/dL")
        if st.sidebar.button("🔄 New profile"):
            st.session_state.profile_created = False
            st.session_state.solid_foods_list = []
            st.session_state.selected_foods_list = []
            st.rerun()

        age_months = calculate_age_months(st.session_state.user_birth_year, st.session_state.user_birth_month, st.session_state.user_birth_day)
        st.title("PKU Meal Planning")
       
        # Baby flow (0-12 months)
        if st.session_state.user_age_category == "Baby (0-12 months)":
            if age_months >= 6:
                st.info("Baby is ≥6 months — you can add solid foods (beikost).")
                with st.expander("🍽️ Add solid foods", expanded=True):
                    meal_type = st.selectbox("Meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                    available_foods = BABY_FOODS if age_months < 9 else {**BABY_FOODS, **TABLE_FOODS}
                    category = st.selectbox("Food category:", list(available_foods.keys()))
                    query = st.text_input("Type to search foods:", "")
                    options = list(available_foods[category].keys())
                    filtered = [o for o in options if query.lower() in o.lower()] if query else options
                    food_name = st.selectbox("Food:", filtered)
                    fd = available_foods[category][food_name]
                    st.write(f"**Standard serving:** {fd['weight_g']} g | {fd['phe_mg']} mg PHE | {fd['protein_g']} g protein | {fd['calories']} kcal")
                    servings = st.number_input("Servings:", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
                    if st.button("➕ Add food"):
                        entry = {
                            "meal": meal_type,
                            "name": food_name,
                            "weight_g": fd['weight_g'] * servings,
                            "phe_mg": fd['phe_mg'] * servings,
                            "protein_g": fd['protein_g'] * servings,
                            "calories": fd['calories'] * servings,
                        }
                        st.session_state.solid_foods_list.append(entry)
                        st.success(f"✅ Added {servings}× {food_name}")
                        st.rerun()

                    if st.session_state.solid_foods_list:
                        st.markdown("---")
                        st.markdown("### Current solid foods")
                        for i, food in enumerate(st.session_state.solid_foods_list):
                            c1, c2, c3 = st.columns([3, 2, 1])
                            with c1:
                                st.write(f"{food['meal']}: {food['name']}")
                            with c2:
                                st.write(f"{food['weight_g']:.0f} g | {food['phe_mg']:.0f} mg PHE")
                            with c3:
                                if st.button("🗑️", key=f"del_baby_{i}"):
                                    st.session_state.solid_foods_list.pop(i)
                                    st.rerun()
                        if st.button("🗑️ Clear all"):
                            st.session_state.solid_foods_list = []
                            st.rerun()

            baby_result = calculate_baby_diet_with_solids(
                age_months,
                st.session_state.user_weight,
                st.session_state.user_milk_type if st.session_state.user_milk_type else "Breast Milk (Human Milk)",
                st.session_state.solid_foods_list
            )
            display_baby_diet_plan(baby_result)

        # Child/Adult flow
        else:
            needs = get_child_adult_daily_needs(age_months, st.session_state.user_weight, st.session_state.user_sex)

            # Add USDA search option
            with st.expander("🔍 Search USDA database for any food"):
                search_query = st.text_input("Search for a food (e.g., 'chicken breast', 'brown rice', 'spinach'):")
                if search_query:
                    with st.spinner("Searching USDA database..."):
                        usda_results = search_usda_foods(search_query)
                    
                    if usda_results:
                        st.success(f"Found {len(usda_results)} results:")
                        for result in usda_results[:10]:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"**{result.get('description')}**")
                                if result.get('brandName'):
                                    st.caption(f"Brand: {result['brandName']}")
                            with col2:
                                if st.button("Select", key=f"usda_{result.get('fdcId')}"):
                                    with st.spinner("Loading nutrition info..."):
                                        details = get_usda_food_details(result.get('fdcId'))
                                    if details is not None:
                                        st.session_state.usda_selected_food = {
                                            "name": result.get('description'),
                                            "details": details
                                        }
                                        st.rerun()
                    elif USDA_API_KEY is not None:
                        st.info("No results found. Try different keywords.")

            # Handle selected USDA food
            if 'usda_selected_food' in st.session_state:
                food = st.session_state.usda_selected_food
                st.success(f"✅ Selected: {food['name']}")
                
                details = food['details']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PHE (per 100g)", f"{details['phe(mg)']:.1f} mg")
                with col2:
                    st.metric("Protein (per 100g)", f"{details['protein(g)']:.1f} g")
                with col3:
                    st.metric("Calories (per 100g)", f"{details['energy(kcal)']:.0f} kcal")
                
                grams = st.number_input("How many grams?", min_value=1.0, value=100.0, step=10.0)
                meal_choice = st.selectbox("Add to meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➕ Add to plan", type="primary"):
                        scaled = scale_nutrients(details, grams)
                        st.session_state.selected_foods_list.append({
                            "meal": meal_choice,
                            "name": food['name'],
                            "weight_g": scaled["weight_g"],
                            "phe_mg": scaled["phe_mg"],
                            "protein_g": scaled["protein_g"],
                            "calories": scaled["calories"],
                        })
                        update_daily_totals()
                        del st.session_state.usda_selected_food
                        st.success(f"Added {grams}g of {food['name']}!")
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        del st.session_state.usda_selected_food
                        st.rerun()

            st.markdown("---")
            cuisine_choice = st.selectbox("Choose cuisine:", ["All Foods"] + list(cuisine_db.keys()))

            if cuisine_choice == "All Foods":
                cuisine_df = consolidated_db
            else:
                cuisine_df = cuisine_db[cuisine_choice]

            cuisine_without_types = cuisine_choice in ["Japanese", "Mediterranean", "Mexican"]
            meal_category = st.radio("Meal category:", ["Breakfast/Snack", "Lunch/Dinner"])
            if not cuisine_without_types and "Meal Type" in cuisine_df.columns and cuisine_df["Meal Type"].nunique() > 1:
                filtered_df = cuisine_df[cuisine_df["Meal Type"] == ("BS" if meal_category == "Breakfast/Snack" else "LD")]
            else:
                filtered_df = cuisine_df

            st.markdown("#### Search and select a dish")
            search_query_dish = st.text_input("Type to search dishes:", "", key="dish_search")
            unique_dishes = filtered_df["dish"].dropna().astype(str).unique().tolist() if "dish" in filtered_df.columns else []
            matching_dishes = [d for d in unique_dishes if search_query_dish.lower() in d.lower()] if search_query_dish else unique_dishes
            selected_dish = st.selectbox("Available dishes:", matching_dishes) if matching_dishes else None

            if selected_dish:
                dish_rows = filtered_df[filtered_df["dish"] == selected_dish] if "dish" in filtered_df.columns else pd.DataFrame()
                dish_nutrients = compute_dish_nutrients(dish_rows, consolidated_db)

                with st.expander(f"📖 View ingredients for '{selected_dish}'", expanded=True):
                    st.markdown("**Ingredients:**")
                    for ing in dish_nutrients["ingredients"]:
                        line = f"- **{ing['name']}**: {ing['weight_g']:.0f} g → "
                        line += f"PHE: {ing['phe_mg']:.1f} mg, Protein: {ing['protein_g']:.2f} g, Calories: {ing['calories']:.0f} kcal"
                        st.write(line)
                        if ing.get("usda_match"):
                            data_type = ing.get("usda_data_type", "")
                            st.caption(f"   ↳ USDA: {ing['usda_match']} [{data_type}]")
                        if ing.get("note"):
                            st.caption(f"   ⚠️ {ing['note']}")
                    st.markdown("---")
                    tot = dish_nutrients["totals"]
                    st.markdown(f"**Dish totals:** PHE {tot['phe_mg']:.1f} mg | Protein {tot['protein_g']:.2f} g | Calories {tot['calories']:.0f} kcal | Weight {tot['weight_g']:.0f} g")
                    st.caption("💡 PHE values calculated from protein content (50 mg PHE per gram protein)")

                assign_meal = st.selectbox("Assign to meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                if st.button(f"➕ Add '{selected_dish}' to my plan", type="primary"):
                    st.session_state.selected_foods_list.append({
                        "meal": assign_meal,
                        "name": selected_dish,
                        "weight_g": dish_nutrients["totals"]["weight_g"],
                        "phe_mg": dish_nutrients["totals"]["phe_mg"],
                        "protein_g": dish_nutrients["totals"]["protein_g"],
                        "calories": dish_nutrients["totals"]["calories"],
                    })
                    update_daily_totals()
                    st.success(f"✅ Added '{selected_dish}' to your plan!")
                    st.rerun()
            elif search_query_dish:
                st.info("No dishes match your search. Try different keywords.")

            st.markdown("---")
            with st.expander("➕ Add a custom dish"):
                custom_dish = add_custom_dish_ui(consolidated_db)
                if custom_dish:
                    st.session_state.selected_foods_list.append({
                        "meal": custom_dish["meal"],
                        "name": custom_dish["name"],
                        "weight_g": custom_dish["weight_g"],
                        "phe_mg": custom_dish["phe_mg"],
                        "protein_g": custom_dish["protein_g"],
                        "calories": custom_dish["calories"],
                    })
                    update_daily_totals()
                    st.rerun()

            if st.session_state.selected_foods_list:
                st.markdown("---")
                st.subheader("📝 Your current meal plan")
                for i, food in enumerate(st.session_state.selected_foods_list):
                    c1, c2, c3 = st.columns([3, 2, 1])
                    with c1:
                        st.write(f"**{food['meal']}:** {food['name']}")
                    with c2:
                        st.write(f"{food['weight_g']:.0f} g | {food['phe_mg']:.0f} mg PHE")
                    with c3:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.selected_foods_list.pop(i)
                            update_daily_totals()
                            st.rerun()
                if st.button("🗑️ Clear all meals"):
                    st.session_state.selected_foods_list = []
                    update_daily_totals()
                    st.rerun()

            total_food_phe = sum(f['phe_mg'] for f in st.session_state.selected_foods_list)
            total_food_protein = sum(f['protein_g'] for f in st.session_state.selected_foods_list)
            total_food_calories = sum(f['calories'] for f in st.session_state.selected_foods_list)

            st.markdown("---")
            st.header("📋 Daily diet plan")
            st.subheader(f"Nutritional targets ({needs['age_group']})")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Protein", f"{needs['protein_g']:.0f} g")
            with c2:
                st.metric("PHE range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
            with c3:
                st.metric("Calories", f"{needs['energy_kcal']:.0f} kcal")

            gap = compute_medical_food_gap(needs['protein_g'], total_food_protein, age_months)
            phe_ok = needs['phe_mg_min'] <= total_food_phe <= needs['phe_mg_max']
            total_protein = total_food_protein + gap['protein_gap_g']
            total_calories = total_food_calories + gap['estimated_calories_kcal']

            st.markdown("---")
            st.markdown("#### Medical food (needs only)")
            st.markdown(f"- Protein gap to fill: **{gap['protein_gap_g']:.1f} g**")
            st.markdown(f"- Estimated powder: {gap['estimated_powder_g']:.1f} g")
            st.markdown(f"- Estimated calories: {gap['estimated_calories_kcal']:.0f} kcal")
            st.markdown("- Phenylalanine: 0 mg (no PHE)")

            st.markdown("---")
            st.subheader("📈 Daily nutrition totals")
            c1, c2, c3 = st.columns(3)
            with c1:
                delta_protein = total_protein - needs['protein_g']
                st.metric("Total protein", f"{total_protein:.1f} g", 
                         f"{'+'if delta_protein >= 0 else ''}{delta_protein:.1f} g",
                         delta_color="normal" if abs(delta_protein) < 5 else "off")
            with c2:
                phe_status = " ✅" if phe_ok else " ⚠️"
                st.metric("Total PHE", f"{total_food_phe:.0f} mg{phe_status}",
                          f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
            with c3:
                delta_cal = total_calories - needs['energy_kcal']
                st.metric("Total calories", f"{total_calories:.0f} kcal", 
                         f"{'+'if delta_cal >= 0 else ''}{delta_cal:.0f} kcal",
                         delta_color="normal" if abs(delta_cal) < 200 else "off")

            remaining_cal = needs['energy_kcal'] - total_calories
            if remaining_cal > 500:
                st.warning(
                    f"⚠️ Additional {remaining_cal:.0f} kcal needed.\n\n"
                    "- Add vegetable oils (120 kcal/Tbsp)\n"
                    "- Low-protein breads and pastas\n"
                    "- PKU-safe fruits and vegetables"
                )
            elif remaining_cal < -500:
                st.warning(
                    f"⚠️ {abs(remaining_cal):.0f} kcal over target.\n\n"
                    "Consider reducing portion sizes or adjusting meal plan."
                )

            if not phe_ok:
                if total_food_phe < needs['phe_mg_min']:
                    st.info(f"ℹ️ PHE is {needs['phe_mg_min'] - total_food_phe:.0f} mg below minimum. Consider adding more protein-containing foods.")
                else:
                    st.warning(f"⚠️ PHE is {total_food_phe - needs['phe_mg_max']:.0f} mg above maximum. Reduce protein-containing foods.")

        st.markdown("---")
        st.header("📖 Important information")
        
        with st.expander("Data attribution"):
            st.markdown(
                "**Recipe data:** Culturally diverse cuisine databases\n\n"
                "**Nutrition data:** USDA FoodData Central - https://fdc.nal.usda.gov\n\n"
                "**Phenylalanine calculation:** PHE content is calculated as 50 mg per gram of protein (5% of total protein), "
                "which is the standard estimation used in PKU dietary management when direct PHE data is unavailable."
            )
        with st.expander("Understanding your numbers"):
            st.markdown(
                "- Children target blood PHE: 2–5 mg/dL; adults: 2–10 mg/dL\n"
                "- Levels should be checked regularly\n"
                "- Adequate energy and protein help stabilize PHE levels"
            )
        with st.expander("⚠️ When to contact your metabolic clinic"):
            st.markdown(
                "- Blood PHE far above target or undetectable\n"
                "- Poor feeding, weight loss, persistent vomiting/diarrhea\n"
                "- Significant behavior changes\n"
                "- Before major diet changes"
            )
        st.warning("⚠️ Important: This app supports planning only. Always follow your metabolic team's recommendations. Never make major diet changes without consulting your doctor/dietitian.")

if __name__ == "__main__":
    main()