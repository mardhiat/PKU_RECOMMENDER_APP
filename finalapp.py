import streamlit as st
import pandas as pd
from datetime import datetime, date
import os
from difflib import SequenceMatcher

# -------------------------------------------------------
# App config
# -------------------------------------------------------
st.set_page_config(page_title="PKU Diet Manager", layout="wide")

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

@st.cache_data
def load_nutritional_data():
    """Load the main nutritional database from Nutritional_Data.csv"""
    try:
        df = pd.read_csv("Nutritional_Data.csv")
        # Standardize column names
        df.columns = [c.strip() for c in df.columns]
        
        # Rename columns to match expected format
        rename_map = {
            "Ingredient": "name",
            "Serving Size (g)": "serving_size(g)",
            "PHE(mg)": "phe(mg)",
            "TYR(mg)": "tyr(mg)",
            "Protein(g)": "protein(g)",
            "Energy(kcal)": "energy(kcal)"
        }
        df = df.rename(columns=rename_map)
        
        # Clean up data
        if "name" in df.columns:
            df["name"] = df["name"].astype(str).str.strip().str.lower()
            # Remove empty rows
            df = df[df["name"].notna() & (df["name"] != "") & (df["name"] != "nan")]
        
        # Convert numeric columns
        for c in ["phe(mg)", "protein(g)", "energy(kcal)", "serving_size(g)", "tyr(mg)"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        
        return df
    except FileNotFoundError:
        st.error("❌ Nutritional_Data.csv not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading nutritional data: {e}")
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

# Load data
nutritional_db = load_nutritional_data()
cuisine_db = load_cuisine_files()

# -------------------------------------------------------
# Smart Search Functions
# -------------------------------------------------------

def normalize_name(s):
    """Normalize ingredient name for matching"""
    return str(s).strip().lower()

def string_similarity(a, b):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def calculate_relevance_score(query, food_name):
    """
    Calculate how relevant a food is to the search query.
    Higher score = better match for the BASE/WHOLE ingredient.
    
    This prioritizes:
    - Whole/raw/fresh foods over processed
    - Simple preparations over complex products
    - Foods where the query is the main subject
    """
    query = normalize_name(query)
    food_name = normalize_name(food_name)
    
    score = 0
    
    # Start with string similarity as base
    similarity = string_similarity(query, food_name)
    score += similarity * 50  # Base similarity score (0-50)
    
    # === BOOST: Whole/raw/base foods ===
    # If the food name IS basically the query (singular/plural match)
    query_variants = [query, query + 's', query + 'es', query.rstrip('s'), query.rstrip('es')]
    for variant in query_variants:
        if food_name == variant:
            score += 100  # Exact base food match
            break
        if food_name.startswith(variant + ','):  # "carrots, fresh" when searching "carrot"
            score += 80
            break
        if food_name.startswith(variant + ' '):  # "apple juice" when searching "apple"  
            score += 40
            break
    
    # Boost for "whole", "raw", "fresh" indicators
    fresh_indicators = ['whole', 'raw', 'fresh', 'plain']
    for indicator in fresh_indicators:
        if indicator in food_name:
            score += 30
    
    # Boost for simple preparations
    simple_preps = ['cooked', 'boiled', 'baked', 'steamed', 'sliced', 'diced', 'mashed', 'chopped']
    for prep in simple_preps:
        if prep in food_name:
            score += 15
    
    # === PENALIZE: Processed/derived products ===
    processed_indicators = [
        'cookie', 'cookies', 'cracker', 'crackers', 'chip', 'chips',
        'cake', 'cakes', 'bar', 'bars', 'candy', 'candies',
        'sauce', 'juice', 'butter', 'jam', 'jelly', 'syrup',
        'dried', 'canned', 'frozen', 'sweetened', 'flavored',
        'mix', 'cereal', 'bread', 'pasta', 'noodle',
        'powder', 'extract', 'concentrate', 'cobbler', 'pie',
        'dessert', 'pudding', 'cream'
    ]
    for indicator in processed_indicators:
        if indicator in food_name:
            score -= 25
    
    # Extra penalty if query is NOT the first word
    # (e.g., "apple cinnamon cookies" when searching "apple" - apple isn't the main subject)
    first_word = food_name.split()[0] if food_name else ""
    first_word = first_word.rstrip(',')
    if query not in [first_word, first_word + 's', first_word.rstrip('s')]:
        # The query isn't the main subject of this food
        if query in food_name:
            score -= 20  # It contains the query but isn't primarily about it
    
    return score

def smart_search_ingredient(query, food_db, top_n=10):
    """
    Smart search that prioritizes base/whole foods over processed items.
    Returns list of (name, row, score) tuples sorted by relevance.
    """
    if food_db.empty or "name" not in food_db.columns:
        return []
    
    query_normalized = normalize_name(query)
    if not query_normalized or query_normalized == "nan":
        return []
    
    results = []
    
    for idx, row in food_db.iterrows():
        food_name = str(row["name"])
        if not food_name or food_name == "nan":
            continue
        
        # Check if this food is related to the query at all
        query_variants = [query_normalized, query_normalized + 's', query_normalized + 'es', 
                         query_normalized.rstrip('s'), query_normalized.rstrip('es')]
        
        is_related = False
        for variant in query_variants:
            if variant in food_name or food_name in variant:
                is_related = True
                break
        
        # Also check fuzzy similarity for typos
        similarity = string_similarity(query_normalized, food_name)
        if similarity > 0.6:
            is_related = True
        
        if is_related:
            score = calculate_relevance_score(query_normalized, food_name)
            # Only include if score is positive (relevant match)
            if score > 0:
                results.append((food_name, row, score))
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:top_n]

def search_ingredient(query, food_db):
    """
    Get the single best match for a query.
    Returns (row, match_info) tuple.
    """
    results = smart_search_ingredient(query, food_db, top_n=1)
    if results:
        name, row, score = results[0]
        return row, f"relevance: {score:.0f}"
    
    # Fallback: try pure fuzzy match for things not caught above
    query_normalized = normalize_name(query)
    best_score = 0
    best_idx = -1
    
    for idx, row in food_db.iterrows():
        food_name = str(row["name"])
        if not food_name or food_name == "nan":
            continue
        similarity = string_similarity(query_normalized, food_name)
        if similarity > best_score:
            best_score = similarity
            best_idx = idx
    
    if best_score > 0.5 and best_idx >= 0:
        return food_db.iloc[best_idx], f"fuzzy ({best_score:.0%})"
    
    return None, "not_found"

def search_foods_list(query, food_db, limit=20):
    """
    Search and return a list of matching foods for display.
    Returns list of (name, row) tuples sorted by relevance.
    """
    if food_db.empty or "name" not in food_db.columns:
        return []
    
    query_normalized = normalize_name(query)
    if not query_normalized:
        # Return first N items if no query
        return [(row["name"], row) for _, row in food_db.head(limit).iterrows()]
    
    # Use smart search to get relevance-sorted results
    smart_results = smart_search_ingredient(query, food_db, top_n=limit)
    
    if smart_results:
        return [(name, row) for name, row, score in smart_results]
    
    # Fallback to simple contains search
    results = []
    seen_names = set()
    
    for _, row in food_db.iterrows():
        name = str(row["name"])
        if query_normalized in name and name not in seen_names:
            results.append((name, row))
            seen_names.add(name)
    
    return results[:limit]

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
    """Calculate daily nutritional needs for infants 0-12 months"""
    needs = {}
    if age_months < 3:
        needs.update({
            'protein_g_per_kg': 3.5, 
            'phe_mg_per_kg_min': 25, 
            'phe_mg_per_kg_max': 70, 
            'energy_kcal_per_kg': 120, 
            'fluid_ml_per_kg': 160, 
            'age_group': '0-3 months'
        })
    elif age_months < 6:
        needs.update({
            'protein_g_per_kg': 3.5, 
            'phe_mg_per_kg_min': 20, 
            'phe_mg_per_kg_max': 45, 
            'energy_kcal_per_kg': 120, 
            'fluid_ml_per_kg': 160, 
            'age_group': '3-6 months'
        })
    elif age_months < 9:
        needs.update({
            'protein_g_per_kg': 3.0, 
            'phe_mg_per_kg_min': 15, 
            'phe_mg_per_kg_max': 35, 
            'energy_kcal_per_kg': 110, 
            'fluid_ml_per_kg': 145, 
            'age_group': '6-9 months'
        })
    else:  # 9-12 months
        needs.update({
            'protein_g_per_kg': 3.0, 
            'phe_mg_per_kg_min': 10, 
            'phe_mg_per_kg_max': 35, 
            'energy_kcal_per_kg': 105, 
            'fluid_ml_per_kg': 135, 
            'age_group': '9-12 months'
        })
    
    # Calculate actual values based on weight
    needs['protein_g'] = needs['protein_g_per_kg'] * weight_kg
    needs['phe_mg_min'] = needs['phe_mg_per_kg_min'] * weight_kg
    needs['phe_mg_max'] = needs['phe_mg_per_kg_max'] * weight_kg
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    needs['energy_kcal'] = needs['energy_kcal_per_kg'] * weight_kg
    needs['fluid_ml'] = needs['fluid_ml_per_kg'] * weight_kg
    return needs

def get_child_adult_daily_needs(age_months, weight_kg, sex):
    """Calculate daily nutritional needs for children and adults"""
    needs = {}
    if age_months < 48:  # 1-4 years
        needs.update({
            'phe_mg_min': 200, 
            'phe_mg_max': 400, 
            'protein_g': 30, 
            'energy_kcal': 1300, 
            'age_group': '1-4 years'
        })
    elif age_months < 84:  # 4-7 years
        needs.update({
            'phe_mg_min': 210, 
            'phe_mg_max': 450, 
            'protein_g': 35, 
            'energy_kcal': 1700, 
            'age_group': '4-7 years'
        })
    elif age_months < 132:  # 7-11 years
        needs.update({
            'phe_mg_min': 220, 
            'phe_mg_max': 500, 
            'protein_g': 40, 
            'energy_kcal': 2400, 
            'age_group': '7-11 years'
        })
    elif age_months < 180:  # 11-15 years
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
    elif age_months < 228:  # 15-19 years
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
    else:  # 19+ years
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
    
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    return needs

def calculate_milk_amount(phe_target_mg, milk_type, split_ratio=0.5):
    """Calculate milk amounts based on PHE target"""
    if milk_type is None:
        milk_type = "Breast Milk (Human Milk)"
    
    if milk_type == "Breast Milk (Human Milk)":
        phe_per_100ml, protein_per_100ml, energy_per_100ml = 48, 1.07, 72
        ml = (phe_target_mg / phe_per_100ml) * 100.0
        return {
            "milk_type": milk_type, 
            "milk_ml": ml,
            "phe_mg": phe_target_mg,
            "protein_g": (ml / 100.0) * protein_per_100ml,
            "calories_kcal": (ml / 100.0) * energy_per_100ml
        }
    elif milk_type == "Similac With Iron":
        phe_per_100ml, protein_per_100ml, energy_per_100ml = 59, 1.40, 68
        ml = (phe_target_mg / phe_per_100ml) * 100.0
        return {
            "milk_type": milk_type, 
            "milk_ml": ml,
            "phe_mg": phe_target_mg,
            "protein_g": (ml / 100.0) * protein_per_100ml,
            "calories_kcal": (ml / 100.0) * energy_per_100ml
        }
    elif "Both" in str(milk_type):
        phe_breast = phe_target_mg * split_ratio
        phe_similac = phe_target_mg * (1 - split_ratio)
        breast = calculate_milk_amount(phe_breast, "Breast Milk (Human Milk)")
        similac = calculate_milk_amount(phe_similac, "Similac With Iron")
        return {
            "milk_type": milk_type,
            "milk_ml": breast["milk_ml"] + similac["milk_ml"],
            "phe_mg": breast["phe_mg"] + similac["phe_mg"],
            "protein_g": breast["protein_g"] + similac["protein_g"],
            "calories_kcal": breast["calories_kcal"] + similac["calories_kcal"],
            "breakdown": {
                "breast_ml": breast["milk_ml"],
                "similac_ml": similac["milk_ml"]
            }
        }
    else:
        return {
            "milk_type": "Breast Milk (Human Milk)", 
            "milk_ml": 0,
            "phe_mg": 0, 
            "protein_g": 0, 
            "calories_kcal": 0
        }

def compute_medical_food_gap(protein_needed_g, protein_from_food_g, age_months):
    """Calculate medical food (formula) needs to fill protein gap"""
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
# Ingredient mapping with local database
# -------------------------------------------------------

def parse_portion(text):
    """Parse portion/amount from text"""
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

def scale_nutrients(row, weight_g):
    """Scale nutrients from serving size to actual weight"""
    serving_size = row.get("serving_size(g)", 100.0)
    if serving_size == 0 or pd.isna(serving_size):
        serving_size = 100.0
    
    phe_serv = row.get("phe(mg)", 0.0)
    prot_serv = row.get("protein(g)", 0.0)
    cal_serv = row.get("energy(kcal)", 0.0)

    phe_per_g = phe_serv / serving_size
    prot_per_g = prot_serv / serving_size
    cal_per_g = cal_serv / serving_size
    
    return {
        "weight_g": weight_g,
        "phe_mg": phe_per_g * weight_g,
        "protein_g": prot_per_g * weight_g,
        "calories": cal_per_g * weight_g,
    }

def compute_dish_nutrients(dish_df, food_db):
    """Compute total nutrients for a dish from its ingredients"""
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
        
        match, match_type = search_ingredient(ing_name, food_db)
        
        if match is None:
            ingredients_list.append({
                "name": ing_name,
                "weight_g": amount,
                "phe_mg": 0.0,
                "protein_g": 0.0,
                "calories": 0.0,
                "note": "Not found in database",
                "match_type": match_type
            })
            total_weight += amount
            continue
        
        scaled = scale_nutrients(match, amount)
        ingredient_entry = {
            "name": ing_name,
            "matched_name": match["name"],
            "match_type": match_type,
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
    """Update session state with current daily totals"""
    totals = {"phe_mg": 0.0, "protein_g": 0.0, "calories": 0.0}
    for food in st.session_state.selected_foods_list:
        totals["phe_mg"] += food.get("phe_mg", 0.0)
        totals["protein_g"] += food.get("protein_g", 0.0)
        totals["calories"] += food.get("calories", 0.0)
    st.session_state.daily_totals = totals

# -------------------------------------------------------
# Baby Diet Functions (FIXED)
# -------------------------------------------------------

def calculate_baby_diet_with_solids(age_months, weight_kg, milk_type, solid_foods, split_ratio=0.5):
    """
    Calculate complete baby diet including milk and solid foods.
    
    FIXED: Now uses the milk_type parameter instead of session state directly.
    """
    needs = get_infant_daily_needs(age_months, weight_kg)
    
    # Calculate totals from solid foods
    total_solid_phe = sum(food['phe_mg'] for food in solid_foods)
    total_solid_protein = sum(food['protein_g'] for food in solid_foods)
    total_solid_calories = sum(food['calories'] for food in solid_foods)
    
    # Calculate remaining PHE budget for milk
    phe_target = needs['phe_mg_max']
    remaining_phe = phe_target - total_solid_phe
    
    if remaining_phe < 0:
        # If solid foods exceed max, try to stay within min
        remaining_phe = max(needs['phe_mg_min'] - total_solid_phe, 0)
    
    # Calculate milk amounts using the passed milk_type parameter
    milk = calculate_milk_amount(remaining_phe, milk_type, split_ratio)
    
    # Check if total exceeds maximum
    total_phe = total_solid_phe + milk['phe_mg']
    if total_phe > needs['phe_mg_max']:
        safe_remaining = max(needs['phe_mg_max'] - total_solid_phe, 0)
        milk = calculate_milk_amount(safe_remaining, milk_type, split_ratio)
    
    # Calculate medical food needs
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
    """Display the baby diet plan with all components"""
    st.markdown("---")
    
    # Milk information
    st.subheader("🍼 Milk Recommendation")
    milk = result['milk']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Milk", f"{milk['milk_ml']:.0f} mL")
    with col2:
        st.metric("From Milk - PHE", f"{milk['phe_mg']:.0f} mg")
    with col3:
        st.metric("From Milk - Protein", f"{milk['protein_g']:.1f} g")
    
    # If using both milk types, show breakdown
    if 'breakdown' in milk:
        st.caption(f"Breast milk: {milk['breakdown']['breast_ml']:.0f} mL | Similac: {milk['breakdown']['similac_ml']:.0f} mL")
    
    # Solid foods if present
    if result['solid_foods']['foods']:
        st.markdown("---")
        st.subheader("🥕 Solid Foods")
        solids = result['solid_foods']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("From Solids - PHE", f"{solids['total_phe_mg']:.0f} mg")
        with col2:
            st.metric("From Solids - Protein", f"{solids['total_protein_g']:.1f} g")
        with col3:
            st.metric("From Solids - Calories", f"{solids['total_calories']:.0f} kcal")
    
    # Medical food
    st.markdown("---")
    st.subheader("💊 Medical Food (PKU Formula)")
    gap = result['medical_food_gap']
    if gap['protein_gap_g'] > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Protein Gap", f"{gap['protein_gap_g']:.1f} g")
        with col2:
            st.metric("Formula Powder", f"{gap['estimated_powder_g']:.1f} g")
        with col3:
            st.metric("Formula Calories", f"{gap['estimated_calories_kcal']:.0f} kcal")
    else:
        st.success("✅ No additional medical food needed - protein needs met by milk and food")
    
    # Daily totals
    st.markdown("---")
    st.subheader("📈 Daily Totals")
    totals = result['totals']
    needs = result['needs']
    
    c1, c2, c3 = st.columns(3)
    with c1:
        protein_diff = totals['protein_g'] - needs['protein_g']
        st.metric(
            "Total Protein", 
            f"{totals['protein_g']:.1f} g", 
            f"Target: {needs['protein_g']:.1f} g"
        )
    with c2:
        in_range = needs['phe_mg_min'] <= totals['phe_mg'] <= needs['phe_mg_max']
        status = " ✅" if in_range else " ⚠️"
        st.metric(
            "Total PHE", 
            f"{totals['phe_mg']:.0f} mg{status}",
            f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg"
        )
    with c3:
        st.metric(
            "Total Calories", 
            f"{totals['calories_kcal']:.0f} kcal", 
            f"Target: {needs['energy_kcal']:.0f} kcal"
        )
    
    # Warnings
    if totals['phe_mg'] > needs['phe_mg_max']:
        st.warning(f"⚠️ PHE is {totals['phe_mg'] - needs['phe_mg_max']:.0f} mg above maximum. Consider reducing solid food portions.")
    elif totals['phe_mg'] < needs['phe_mg_min']:
        st.info(f"ℹ️ PHE is {needs['phe_mg_min'] - totals['phe_mg']:.0f} mg below minimum. Baby may need more natural protein.")

def add_custom_dish_ui(food_db):
    """UI for adding custom dishes with ingredients"""
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
        
        dish_nutrients = compute_dish_nutrients(dish_df, food_db)
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
    # Initialize session state
    if 'profile_created' not in st.session_state:
        st.session_state.profile_created = False
    if 'solid_foods_list' not in st.session_state:
        st.session_state.solid_foods_list = []
    if 'selected_foods_list' not in st.session_state:
        st.session_state.selected_foods_list = []

    # Check if database loaded
    if nutritional_db.empty:
        st.error("❌ Could not load nutritional database. Please ensure Nutritional_Data.csv is in the app directory.")
        st.stop()
    
    # ==========================================
    # PROFILE CREATION
    # ==========================================
    if not st.session_state.profile_created:
        st.title("PKU Diet Management System")
        st.markdown("Plan safe PKU diets with comprehensive nutrition tracking and medical food calculations.")
        st.info(f"🥗 Database contains {len(nutritional_db)} PKU-safe foods")
        st.markdown("---")
        st.header("Create Profile")

        age_category = st.radio("Profile type:", ["Baby (0-12 months)", "Child (1-12 years)", "Adult (12+ years)"])
        sex = st.radio("Sex:", ["Male", "Female"]) if age_category != "Baby (0-12 months)" else "Male"

        col1, col2 = st.columns(2)
        with col1:
            units = st.radio("Units:", ["Metric", "Imperial"])
            if units == "Metric":
                weight = st.number_input(
                    'Weight (kg):', 
                    min_value=0.0, 
                    step=0.1,
                    value=7.0 if age_category == "Baby (0-12 months)" else 20.0
                )
                height_cm = st.number_input(
                    'Height (cm):', 
                    min_value=0.0, 
                    step=1.0,
                    value=65.0 if age_category == "Baby (0-12 months)" else 120.0
                )
            else:
                weight_lbs = st.number_input(
                    'Weight (lbs):', 
                    min_value=0.0, 
                    step=0.1,
                    value=15.4 if age_category == "Baby (0-12 months)" else 44.0
                )
                height_in = st.number_input(
                    'Height (in):', 
                    min_value=0.0, 
                    step=0.5,
                    value=25.6 if age_category == "Baby (0-12 months)" else 47.0
                )
                weight = weight_lbs * 0.453592
                height_cm = height_in * 2.54
                
        with col2:
            birth_year = st.number_input(
                'Birth year:', 
                min_value=1900, 
                max_value=datetime.now().year,
                value=2024 if age_category == "Baby (0-12 months)" else 2017
            )
            birth_month = st.number_input(
                'Birth month:', 
                min_value=1, 
                max_value=12,
                value=6 if age_category == "Baby (0-12 months)" else 1
            )
            birth_day = st.number_input('Birth day:', min_value=1, max_value=31, value=1)
            current_phe = st.number_input('Current blood PHE (mg/dL):', min_value=0.0, step=0.1, value=5.0)

        milk_type = None
        milk_split_ratio = 0.5
        if age_category == "Baby (0-12 months)":
            milk_type = st.radio("Milk type:", ["Breast Milk (Human Milk)", "Similac With Iron", "Both"])
            
            if milk_type == "Both":
                milk_split_ratio = st.slider("Proportion breast milk vs. Similac", 0.0, 1.0, 0.5)
                milk_type = f"Both (Breast {milk_split_ratio*100:.0f}%, Similac {(1-milk_split_ratio)*100:.0f}%)"

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
                st.session_state.milk_split_ratio = milk_split_ratio
                st.rerun()
    
    # ==========================================
    # MAIN APP (AFTER PROFILE CREATED)
    # ==========================================
    else:
        # Sidebar profile
        st.sidebar.header("👤 Profile")
        bdate = date(
            int(st.session_state.user_birth_year), 
            int(st.session_state.user_birth_month), 
            int(st.session_state.user_birth_day)
        )
        st.sidebar.write(f"**Age:** {format_age_display(bdate)}")
        st.sidebar.write(f"**Weight:** {st.session_state.user_weight:.1f} kg")
        st.sidebar.write(f"**Height:** {st.session_state.user_height_cm:.1f} cm")
        st.sidebar.write(f"**Current PHE:** {st.session_state.user_current_phe:.1f} mg/dL")
        
        if st.sidebar.button("🔄 New profile"):
            st.session_state.profile_created = False
            st.session_state.solid_foods_list = []
            st.session_state.selected_foods_list = []
            st.rerun()

        age_months = calculate_age_months(
            st.session_state.user_birth_year, 
            st.session_state.user_birth_month, 
            st.session_state.user_birth_day
        )
        
        st.title("PKU Meal Planning")
       
        # ==========================================
        # BABY FLOW (0-12 months)
        # ==========================================
        if st.session_state.user_age_category == "Baby (0-12 months)":
            st.info(f"👶 Baby is {age_months} months old")
            
            if age_months >= 6:
                st.success("Baby is ≥6 months — you can add solid foods (beikost).")
                
                with st.expander("🍽️ Add solid foods", expanded=True):
                    meal_type = st.selectbox("Meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                    
                    # Choose between preset foods or search database
                    food_source = st.radio("Food source:", ["Preset Baby Foods", "Search Database"])
                    
                    if food_source == "Preset Baby Foods":
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
                    
                    else:  # Search Database
                        search_query = st.text_input("Search for food:", "")
                        if search_query:
                            results = search_foods_list(search_query, nutritional_db, limit=10)
                            if results:
                                food_options = [f"{name} (PHE: {row['phe(mg)']:.0f}mg per {row['serving_size(g)']:.0f}g)" 
                                              for name, row in results]
                                selected_idx = st.selectbox("Select food:", range(len(food_options)), 
                                                          format_func=lambda x: food_options[x])
                                
                                if selected_idx is not None:
                                    selected_name, selected_row = results[selected_idx]
                                    serving = selected_row['serving_size(g)']
                                    grams = st.number_input("Amount (grams):", min_value=1.0, value=float(serving), step=5.0)
                                    
                                    scaled = scale_nutrients(selected_row, grams)
                                    st.write(f"**Nutrition for {grams:.0f}g:** PHE: {scaled['phe_mg']:.1f} mg | Protein: {scaled['protein_g']:.2f} g | Calories: {scaled['calories']:.0f}")
                                    
                                    if st.button("➕ Add to meal"):
                                        entry = {
                                            "meal": meal_type,
                                            "name": selected_name,
                                            "weight_g": grams,
                                            "phe_mg": scaled['phe_mg'],
                                            "protein_g": scaled['protein_g'],
                                            "calories": scaled['calories'],
                                        }
                                        st.session_state.solid_foods_list.append(entry)
                                        st.success(f"✅ Added {grams}g of {selected_name}")
                                        st.rerun()
                            else:
                                st.info("No matching foods found. Try different keywords.")

                    # Display current solid foods
                    if st.session_state.solid_foods_list:
                        st.markdown("---")
                        st.markdown("### Current solid foods")
                        for i, food in enumerate(st.session_state.solid_foods_list):
                            c1, c2, c3 = st.columns([3, 2, 1])
                            with c1:
                                st.write(f"**{food['meal']}:** {food['name']}")
                            with c2:
                                st.write(f"{food['weight_g']:.0f}g | {food['phe_mg']:.0f}mg PHE")
                            with c3:
                                if st.button("🗑️", key=f"del_baby_{i}"):
                                    st.session_state.solid_foods_list.pop(i)
                                    st.rerun()
                        
                        if st.button("🗑️ Clear all solid foods"):
                            st.session_state.solid_foods_list = []
                            st.rerun()
            else:
                st.info("👶 Baby is under 6 months - diet is milk only. Solid foods can be introduced at 6 months.")

            # Calculate and display baby diet plan
            baby_result = calculate_baby_diet_with_solids(
                age_months,
                st.session_state.user_weight,
                st.session_state.user_milk_type if st.session_state.user_milk_type else "Breast Milk (Human Milk)",
                st.session_state.solid_foods_list,
                st.session_state.get("milk_split_ratio", 0.5)
            )
            display_baby_diet_plan(baby_result)

        # ==========================================
        # CHILD/ADULT FLOW
        # ==========================================
        else:
            needs = get_child_adult_daily_needs(age_months, st.session_state.user_weight, st.session_state.user_sex)

            # Food search section
            with st.expander("🔍 Search database for any food", expanded=True):
                search_query = st.text_input("Search for a food (e.g., 'apple', 'rice', 'carrot'):", key="main_search")
                
                if search_query:
                    results = search_foods_list(search_query, nutritional_db, limit=15)
                    
                    if results:
                        st.success(f"Found {len(results)} matching foods:")
                        
                        for idx, (name, row) in enumerate(results):
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.write(f"**{name.title()}**")
                            with col2:
                                st.caption(f"PHE: {row['phe(mg)']:.0f}mg | Protein: {row['protein(g)']:.1f}g per {row['serving_size(g)']:.0f}g")
                            with col3:
                                if st.button("Select", key=f"select_{idx}"):
                                    st.session_state.selected_search_food = {
                                        "name": name,
                                        "row": row
                                    }
                                    st.rerun()
                    else:
                        st.info("No matching foods found. Try different keywords.")

            # Handle selected food
            if 'selected_search_food' in st.session_state:
                food = st.session_state.selected_search_food
                st.success(f"✅ Selected: {food['name'].title()}")
                
                row = food['row']
                serving = row['serving_size(g)']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PHE", f"{row['phe(mg)']:.1f} mg")
                with col2:
                    st.metric("Protein", f"{row['protein(g)']:.2f} g")
                with col3:
                    st.metric("Calories", f"{row['energy(kcal)']:.0f} kcal")
                with col4:
                    st.metric("Serving", f"{serving:.0f} g")
                
                grams = st.number_input("How many grams?", min_value=1.0, value=float(serving), step=10.0)
                meal_choice = st.selectbox("Add to meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                
                scaled = scale_nutrients(row, grams)
                st.info(f"**For {grams:.0f}g:** PHE: {scaled['phe_mg']:.1f} mg | Protein: {scaled['protein_g']:.2f} g | Calories: {scaled['calories']:.0f} kcal")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➕ Add to plan", type="primary"):
                        st.session_state.selected_foods_list.append({
                            "meal": meal_choice,
                            "name": food['name'].title(),
                            "weight_g": grams,
                            "phe_mg": scaled["phe_mg"],
                            "protein_g": scaled["protein_g"],
                            "calories": scaled["calories"],
                        })
                        update_daily_totals()
                        del st.session_state.selected_search_food
                        st.success(f"Added {grams}g of {food['name'].title()}!")
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        del st.session_state.selected_search_food
                        st.rerun()

            # Cuisine-based dishes
            st.markdown("---")
            if cuisine_db:
                cuisine_choice = st.selectbox("Or choose a cuisine for pre-made dishes:", ["None"] + list(cuisine_db.keys()))

                if cuisine_choice != "None":
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
                        dish_nutrients = compute_dish_nutrients(dish_rows, nutritional_db)

                        with st.expander(f"📖 View ingredients for '{selected_dish}'", expanded=True):
                            st.markdown("**Ingredients:**")
                            for ing in dish_nutrients["ingredients"]:
                                line = f"- **{ing['name']}**: {ing['weight_g']:.0f} g → "
                                line += f"PHE: {ing['phe_mg']:.1f} mg, Protein: {ing['protein_g']:.2f} g, Calories: {ing['calories']:.0f} kcal"
                                st.write(line)
                                if ing.get("matched_name") and ing.get("matched_name") != ing['name'].lower():
                                    st.caption(f"   ↳ Matched to: {ing['matched_name']} ({ing.get('match_type', 'unknown')})")
                                if ing.get("note"):
                                    st.caption(f"   ⚠️ {ing['note']}")
                            st.markdown("---")
                            tot = dish_nutrients["totals"]
                            st.markdown(f"**Dish totals:** PHE {tot['phe_mg']:.1f} mg | Protein {tot['protein_g']:.2f} g | Calories {tot['calories']:.0f} kcal | Weight {tot['weight_g']:.0f} g")

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

            # Custom dish
            st.markdown("---")
            with st.expander("➕ Add a custom dish"):
                custom_dish = add_custom_dish_ui(nutritional_db)
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

            # Current meal plan
            if st.session_state.selected_foods_list:
                st.markdown("---")
                st.subheader("📝 Your Current Meal Plan")
                for i, food in enumerate(st.session_state.selected_foods_list):
                    c1, c2, c3 = st.columns([3, 2, 1])
                    with c1:
                        st.write(f"**{food['meal']}:** {food['name']}")
                    with c2:
                        st.write(f"{food['weight_g']:.0f}g | {food['phe_mg']:.0f}mg PHE | {food['protein_g']:.1f}g protein")
                    with c3:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.selected_foods_list.pop(i)
                            update_daily_totals()
                            st.rerun()
                if st.button("🗑️ Clear all meals"):
                    st.session_state.selected_foods_list = []
                    update_daily_totals()
                    st.rerun()

            # Calculate totals
            total_food_phe = sum(f['phe_mg'] for f in st.session_state.selected_foods_list)
            total_food_protein = sum(f['protein_g'] for f in st.session_state.selected_foods_list)
            total_food_calories = sum(f['calories'] for f in st.session_state.selected_foods_list)

            # Daily diet plan summary
            st.markdown("---")
            st.header("📋 Daily Diet Plan")
            st.subheader(f"Nutritional Targets ({needs['age_group']})")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Protein Target", f"{needs['protein_g']:.0f} g")
            with c2:
                st.metric("PHE Range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
            with c3:
                st.metric("Calorie Target", f"{needs['energy_kcal']:.0f} kcal")

            # Medical food calculation
            gap = compute_medical_food_gap(needs['protein_g'], total_food_protein, age_months)
            phe_ok = needs['phe_mg_min'] <= total_food_phe <= needs['phe_mg_max']
            total_protein = total_food_protein + gap['protein_gap_g']
            total_calories = total_food_calories + gap['estimated_calories_kcal']

            st.markdown("---")
            st.markdown("#### 💊 Medical Food (PKU Formula)")
            if gap['protein_gap_g'] > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Protein Gap", f"{gap['protein_gap_g']:.1f} g")
                with col2:
                    st.metric("Formula Powder", f"{gap['estimated_powder_g']:.1f} g")
                with col3:
                    st.metric("Formula Calories", f"{gap['estimated_calories_kcal']:.0f} kcal")
                st.caption("Note: Medical food contains 0 mg PHE")
            else:
                st.success("✅ Protein needs met by food - no additional medical food needed")

            # Final totals
            st.markdown("---")
            st.subheader("📈 Daily Nutrition Totals")
            c1, c2, c3 = st.columns(3)
            with c1:
                delta_protein = total_protein - needs['protein_g']
                st.metric(
                    "Total Protein", 
                    f"{total_protein:.1f} g", 
                    f"{'+'if delta_protein >= 0 else ''}{delta_protein:.1f} g vs target",
                    delta_color="normal" if abs(delta_protein) < 5 else "off"
                )
            with c2:
                phe_status = " ✅" if phe_ok else " ⚠️"
                st.metric(
                    "Total PHE", 
                    f"{total_food_phe:.0f} mg{phe_status}",
                    f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg"
                )
            with c3:
                delta_cal = total_calories - needs['energy_kcal']
                st.metric(
                    "Total Calories", 
                    f"{total_calories:.0f} kcal", 
                    f"{'+'if delta_cal >= 0 else ''}{delta_cal:.0f} kcal vs target",
                    delta_color="normal" if abs(delta_cal) < 200 else "off"
                )

            # Warnings and recommendations
            remaining_cal = needs['energy_kcal'] - total_calories
            if remaining_cal > 500:
                st.warning(
                    f"⚠️ Additional {remaining_cal:.0f} kcal needed.\n\n"
                    "Consider adding:\n"
                    "- Vegetable oils (120 kcal/Tbsp)\n"
                    "- Low-protein breads and pastas\n"
                    "- PKU-safe fruits and vegetables"
                )
            elif remaining_cal < -500:
                st.warning(
                    f"⚠️ {abs(remaining_cal):.0f} kcal over target.\n\n"
                    "Consider reducing portion sizes."
                )

            if not phe_ok:
                if total_food_phe < needs['phe_mg_min']:
                    st.info(f"ℹ️ PHE is {needs['phe_mg_min'] - total_food_phe:.0f} mg below minimum. Consider adding more protein-containing foods.")
                else:
                    st.error(f"⚠️ PHE is {total_food_phe - needs['phe_mg_max']:.0f} mg above maximum! Reduce high-protein foods.")

        # ==========================================
        # INFORMATION SECTION
        # ==========================================
        st.markdown("---")
        st.header("📖 Important Information")
        
        with st.expander("About this app"):
            st.markdown(
                "This PKU Diet Management System helps plan safe, balanced diets for individuals with Phenylketonuria (PKU).\n\n"
                f"**Database:** {len(nutritional_db)} PKU-safe foods with PHE, protein, and calorie data\n\n"
                "**Features:**\n"
                "- Smart food search with fuzzy matching\n"
                "- Baby diet planning (0-12 months) with milk calculations\n"
                "- Child/Adult planning with medical food calculations\n"
                "- Multi-cuisine dish support"
            )
        
        with st.expander("Understanding your numbers"):
            st.markdown(
                "**Blood PHE targets:**\n"
                "- Children: 2-5 mg/dL\n"
                "- Adults: 2-10 mg/dL\n\n"
                "**Key points:**\n"
                "- Levels should be checked regularly\n"
                "- Adequate energy and protein help stabilize PHE levels\n"
                "- Medical food (PKU formula) provides protein without PHE"
            )
        
        with st.expander("⚠️ When to contact your metabolic clinic"):
            st.markdown(
                "Contact your metabolic team if:\n"
                "- Blood PHE is far above target or undetectable\n"
                "- Poor feeding, weight loss, persistent vomiting/diarrhea\n"
                "- Significant behavior changes\n"
                "- Before making major diet changes"
            )
        
        st.warning(
            "⚠️ **Important:** This app is for planning support only. "
            "Always follow your metabolic team's recommendations. "
            "Never make major diet changes without consulting your doctor or dietitian."
        )

if __name__ == "__main__":
    main()