import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import math
import os
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# CORE CALCULATION FUNCTIONS (Based on Protocol Document)

def calculate_age_months(birth_year, birth_month, birth_day):
    """Calculate age in months from birth date"""
    today = datetime.today()
    return (today.year - birth_year) * 12 + (today.month - birth_month)

def calculate_bmi(weight_kg, height_m):
    """Calculate BMI"""
    if height_m == 0:
        return 0
    return weight_kg / (height_m * height_m)

def get_phe_deletion_protocol(phe_mg_dl):
    """
    Determine PHE deletion hours based on diagnostic PHE level
    Protocol Section VII.3, page 3
    """
    if phe_mg_dl < 4:
        return 0, "No deletion needed - PHE level is acceptable"
    elif 4 <= phe_mg_dl < 10:
        return 24, "Monitor daily, add PHE when level reaches 5 mg/dL"
    elif 10 <= phe_mg_dl < 20:
        return 48, "Monitor daily, add PHE when level reaches 5 mg/dL"
    elif 20 <= phe_mg_dl < 40:
        return 72, "Monitor daily, add PHE when level reaches 5 mg/dL"
    else:  # >= 40
        return 96, "Monitor daily, add PHE when level reaches 5 mg/dL"

def get_initial_phe_dose(diagnostic_phe_mg_dl, weight_kg):
    """
    Calculate initial PHE mg/kg after deletion period
    Protocol Section VII.4, page 3-4
    """
    if diagnostic_phe_mg_dl <= 10:
        mg_per_kg = 70
    elif 10 < diagnostic_phe_mg_dl <= 20:
        mg_per_kg = 55
    elif 20 < diagnostic_phe_mg_dl <= 30:
        mg_per_kg = 45
    elif 30 < diagnostic_phe_mg_dl <= 40:
        mg_per_kg = 35
    else:  # > 40
        mg_per_kg = 25
    
    return mg_per_kg * weight_kg, mg_per_kg

def get_infant_daily_needs(age_months, weight_kg):
    """
    Get daily nutritional needs for infants 0-12 months
    From Table 1-1, page 12
    """
    needs = {}
    
    # Determine age bracket
    if age_months < 3:
        needs['protein_g_per_kg'] = 3.5
        needs['phe_mg_per_kg_min'] = 25
        needs['phe_mg_per_kg_max'] = 70
        needs['energy_kcal_per_kg'] = 120
        needs['fluid_ml_per_kg'] = 160
        needs['age_group'] = '0-3 months'
    elif age_months < 6:
        needs['protein_g_per_kg'] = 3.5
        needs['phe_mg_per_kg_min'] = 20
        needs['phe_mg_per_kg_max'] = 45
        needs['energy_kcal_per_kg'] = 120
        needs['fluid_ml_per_kg'] = 160
        needs['age_group'] = '3-6 months'
    elif age_months < 9:
        needs['protein_g_per_kg'] = 3.0
        needs['phe_mg_per_kg_min'] = 15
        needs['phe_mg_per_kg_max'] = 35
        needs['energy_kcal_per_kg'] = 110
        needs['fluid_ml_per_kg'] = 145
        needs['age_group'] = '6-9 months'
    else:  # 9-12 months
        needs['protein_g_per_kg'] = 3.0
        needs['phe_mg_per_kg_min'] = 10
        needs['phe_mg_per_kg_max'] = 35
        needs['energy_kcal_per_kg'] = 105
        needs['fluid_ml_per_kg'] = 135
        needs['age_group'] = '9-12 months'
    
    # Calculate total daily needs
    needs['protein_g'] = needs['protein_g_per_kg'] * weight_kg
    needs['phe_mg_min'] = needs['phe_mg_per_kg_min'] * weight_kg
    needs['phe_mg_max'] = needs['phe_mg_per_kg_max'] * weight_kg
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    needs['energy_kcal'] = needs['energy_kcal_per_kg'] * weight_kg
    needs['fluid_ml'] = needs['fluid_ml_per_kg'] * weight_kg
    
    return needs

def get_child_adult_daily_needs(age_months, weight_kg, sex):
    """
    Get daily nutritional needs for children and adults
    From Table 1-1, page 12
    """
    needs = {}
    
    if age_months < 48:  # 1-4 years
        needs['phe_mg_min'] = 200
        needs['phe_mg_max'] = 400
        needs['protein_g'] = 30
        needs['energy_kcal'] = 1300
        needs['age_group'] = '1-4 years'
    elif age_months < 84:  # 4-7 years
        needs['phe_mg_min'] = 210
        needs['phe_mg_max'] = 450
        needs['protein_g'] = 35
        needs['energy_kcal'] = 1700
        needs['age_group'] = '4-7 years'
    elif age_months < 132:  # 7-11 years
        needs['phe_mg_min'] = 220
        needs['phe_mg_max'] = 500
        needs['protein_g'] = 40
        needs['energy_kcal'] = 2400
        needs['age_group'] = '7-11 years'
    elif age_months < 180:  # 11-15 years
        if sex == "Female":
            needs['phe_mg_min'] = 250
            needs['phe_mg_max'] = 750
            needs['protein_g'] = 50
            needs['energy_kcal'] = 2200
        else:  # Male
            needs['phe_mg_min'] = 225
            needs['phe_mg_max'] = 900
            needs['protein_g'] = 55
            needs['energy_kcal'] = 2700
        needs['age_group'] = '11-15 years'
    elif age_months < 228:  # 15-19 years
        if sex == "Female":
            needs['phe_mg_min'] = 230
            needs['phe_mg_max'] = 700
            needs['protein_g'] = 55
            needs['energy_kcal'] = 2100
        else:  # Male
            needs['phe_mg_min'] = 295
            needs['phe_mg_max'] = 1100
            needs['protein_g'] = 65
            needs['energy_kcal'] = 2800
        needs['age_group'] = '15-19 years'
    else:  # 19+ years
        if sex == "Female":
            needs['phe_mg_min'] = 220
            needs['phe_mg_max'] = 700
            needs['protein_g'] = 60
            needs['energy_kcal'] = 2100
        else:  # Male
            needs['phe_mg_min'] = 290
            needs['phe_mg_max'] = 1200
            needs['protein_g'] = 70
            needs['energy_kcal'] = 2900
        needs['age_group'] = '19+ years'
    
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    
    return needs

def calculate_phenex_amount(protein_needed_g, protein_from_food_g, age_months):
    """
    Calculate grams of Phenex powder needed
    From Table 1-4, page 27
    
    Phenex-1 (0-24 months): 15g protein per 100g powder
    Phenex-2 (24+ months): 30g protein per 100g powder
    """
    protein_from_phenex = protein_needed_g - protein_from_food_g
    
    if age_months < 24:
        # Phenex-1
        phenex_grams = (protein_from_phenex * 100) / 15
        phenex_type = "Phenex-1"
        protein_per_100g = 15
        calories_per_100g = 480
    else:
        # Phenex-2
        phenex_grams = (protein_from_phenex * 100) / 30
        phenex_type = "Phenex-2"
        protein_per_100g = 30
        calories_per_100g = 410
    
    # Calculate what this Phenex provides
    actual_protein = (phenex_grams / 100) * protein_per_100g
    calories = (phenex_grams / 100) * calories_per_100g
    
    return {
        'phenex_type': phenex_type,
        'phenex_grams': phenex_grams,
        'protein_g': actual_protein,
        'calories_kcal': calories
    }

def calculate_formula_amount(phe_target_mg):
    """
    Calculate formula amount based on PHE target
    From Table 1-2, page 12
    
    Similac With Iron (per 100 mL):
    - PHE: 59 mg
    - Protein: 1.40 g
    - Energy: 68 kcal
    """
    formula_ml = (phe_target_mg / 59) * 100
    
    return {
        'formula_ml': formula_ml,
        'phe_mg': phe_target_mg,
        'protein_g': (formula_ml / 100) * 1.40,
        'calories_kcal': (formula_ml / 100) * 68
    }

# BABY DIET CALCULATOR (0-12 months)

def calculate_baby_diet(age_months, weight_kg, current_phe_mg_dl, age_hours=None):
    """
    Complete diet calculation for babies 0-12 months
    """
    result = {}
    
    # Get daily nutritional needs
    needs = get_infant_daily_needs(age_months, weight_kg)
    result['needs'] = needs
    
    # Check if PHE deletion is needed (only for babies ≤96 hours old)
    result['deletion_needed'] = False
    if age_hours is not None and age_hours <= 96 and current_phe_mg_dl >= 4:
        deletion_hours, deletion_note = get_phe_deletion_protocol(current_phe_mg_dl)
        result['deletion_needed'] = True
        result['deletion_hours'] = deletion_hours
        result['deletion_note'] = deletion_note
        
        # During deletion: Phenex-1 ONLY (no formula)
        phenex_deletion = calculate_phenex_amount(needs['protein_g'], 0, age_months)
        result['deletion_phase'] = {
            'phenex': phenex_deletion,
            'water_ml': needs['fluid_ml'],
            'feedings_per_day': '6-8' if age_months < 6 else '4-6'
        }
        
        # After deletion: Calculate maintenance diet
        initial_phe_mg, phe_mg_per_kg = get_initial_phe_dose(current_phe_mg_dl, weight_kg)
        result['initial_phe_mg_per_kg'] = phe_mg_per_kg
        result['initial_phe_mg'] = initial_phe_mg
    else:
        # Use target PHE range for ongoing management
        initial_phe_mg = needs['phe_mg_target']
        result['initial_phe_mg'] = initial_phe_mg
    
    # Calculate formula amount (based on PHE target)
    formula = calculate_formula_amount(initial_phe_mg)
    result['formula'] = formula
    
    # Calculate Phenex amount (remaining protein after formula)
    phenex = calculate_phenex_amount(needs['protein_g'], formula['protein_g'], age_months)
    result['phenex'] = phenex
    
    # Calculate total volume and water needed
    total_volume_needed = needs['fluid_ml']
    water_ml = total_volume_needed - formula['formula_ml']
    result['water_ml'] = water_ml
    result['total_volume_ml'] = total_volume_needed
    
    # Feeding schedule
    result['feedings_per_day'] = '6-8' if age_months < 6 else '4-6'
    avg_feedings = 7 if age_months < 6 else 5
    result['ml_per_feeding'] = total_volume_needed / avg_feedings
    
    # Total nutrition provided
    result['totals'] = {
        'protein_g': formula['protein_g'] + phenex['protein_g'],
        'phe_mg': formula['phe_mg'],
        'calories_kcal': formula['calories_kcal'] + phenex['calories_kcal']
    }
    
    return result


# FILE LOADING FUNCTIONS

def prepare_meal_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare meal dataframe with consistent column names"""
    rename_map = {}
    for col in raw_df.columns:
        lc = col.lower().strip()
        if lc in ['mealtype','meal type','type']: rename_map[col] = 'MealType'
        if 'gram' in lc: rename_map[col] = 'Weight'
        if 'serving' in lc or 'total' in lc: rename_map[col] = 'DishWeight'
        if lc == 'ingredient': rename_map[col] = 'Ingredient'
        if col not in rename_map and any(x in lc for x in ['meal','dish','name']): rename_map[col] = 'Meal'
    
    df = raw_df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'Meal' not in df.columns:
        raise KeyError("No 'Meal' column detected; cannot proceed.")
    
    df['MealGroup'] = df['Meal'].ffill()
    
    if 'MealType' not in df.columns:
        df['MealType'] = 'All'
    
    if 'IngredientsCount' not in df.columns:
        df['IngredientsCount'] = df.groupby('MealGroup')['MealGroup'].transform('size')
    
    if 'DishWeight' not in df.columns and 'Weight' in df.columns:
        df['DishWeight'] = df.groupby('MealGroup')['Weight'].transform('sum')
    
    return df

def load_cuisine_files():
    """Load all cuisine CSV files"""
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
    
    loaded_cuisines = {}
    for name, filename in cuisine_files.items():
        try:
            df = pd.read_csv(filename, encoding='latin1')
            loaded_cuisines[name] = prepare_meal_df(df)
        except FileNotFoundError:
            st.warning(f"File {filename} not found in directory.")
        except Exception as e:
            st.warning(f"Error loading {filename}: {e}")
    
    return loaded_cuisines

def load_ingredient_files():
    """Load ingredient database files"""
    try:
        nutritional_df = pd.read_csv('Nutritional_Data.csv')
        chat_df = pd.read_csv('consolidated_chat_ingredients.csv')
        return nutritional_df, chat_df
    except FileNotFoundError as e:
        st.error(f"Required ingredient file not found: {e}")
        return None, None


# MEAL NUTRITION CALCULATION

def meal_nutrition(dish, serving_size, cuisine_file, ingredients_file):
    """
    Calculate nutrition for a specific dish
    Returns: [protein_g, phe_mg, calories_kcal]
    """
    dish_row = cuisine_file[cuisine_file['Meal'].str.lower() == dish.lower()]
    if dish_row.empty:
        raise ValueError(f"Dish '{dish}' not found in cuisine file")
    
    # Get number of ingredients
    num_ingredients_col = None
    for col in ['Number of Ingredients', 'IngredientsCount', 'Ingredients Count']:
        if col in dish_row.columns:
            num_ingredients_col = col
            break
    
    if num_ingredients_col is None:
        numIngredients = len(cuisine_file[cuisine_file.index >= dish_row.index[0]]) - 1
    else:
        numIngredients = int(dish_row.iloc[0][num_ingredients_col])

    dishProtein = 0
    dishPhenyl = 0
    dishCalories = 0

    for i in range(dish_row.index[0], min(dish_row.index[0] + numIngredients, len(cuisine_file))):
        if i >= len(cuisine_file):
            break
            
        ingredient = cuisine_file.iloc[i]['Ingredient']
        if pd.isna(ingredient):
            continue
            
        ingredient_row = ingredients_file[ingredients_file['Ingredient'].str.lower() == ingredient.lower()]
        if ingredient_row.empty:
            continue

        try:
            grams = float(ingredient_row.iloc[0].iloc[1])
            gramsWanted = float(cuisine_file.iloc[i].iloc[3] if len(cuisine_file.iloc[i]) > 3 else 100)
            
            protein_col = next((col for col in ingredient_row.columns if 'protein' in col.lower()), None)
            phe_col = next((col for col in ingredient_row.columns if 'phe' in col.lower()), None)
            cal_col = next((col for col in ingredient_row.columns if any(x in col.lower() for x in ['energy', 'calor', 'kcal'])), None)
            
            if protein_col and phe_col and cal_col:
                gramsOfProtein = gramsWanted / grams * float(ingredient_row.iloc[0][protein_col])
                gramsOfPhenyl = gramsWanted / grams * float(ingredient_row.iloc[0][phe_col])
                calories = gramsWanted / grams * float(ingredient_row.iloc[0][cal_col])
                
                dishProtein += gramsOfProtein
                dishPhenyl += gramsOfPhenyl
                dishCalories += calories
        except (ValueError, IndexError):
            continue

    try:
        meal_weight_col = next((col for col in dish_row.columns if any(x in col.lower() for x in ['weight', 'grams', 'serving'])), None)
        if meal_weight_col:
            mealGrams = float(dish_row.iloc[0][meal_weight_col])
        else:
            mealGrams = 100
        
        conversion = serving_size / mealGrams
        dishProtein *= conversion
        dishPhenyl *= conversion
        dishCalories *= conversion
    except (ValueError, KeyError):
        pass

    return [dishProtein, dishPhenyl, dishCalories]

# COLLABORATIVE FILTERING FUNCTIONS

def calculate_w_ij(df, dish, ingredient):
    """Calculate weight of ingredient in dish"""
    numMeals = len(df['Meal'].dropna())
    dish_row = df[df['Meal'].str.lower()==dish.lower()]
    if dish_row.empty: return 0.0
    numIngredients = int(dish_row.iloc[0]['IngredientsCount'])
    ing_list = df.iloc[dish_row.index[0]:dish_row.index[0]+numIngredients]
    meals_with = df[df['Ingredient'].str.lower()==ingredient.lower()]
    dishWeight = float(dish_row.iloc[0]['DishWeight'])
    ingrWeight = float(ing_list[ing_list['Ingredient'].str.lower()==ingredient.lower()]['Weight'].iloc[0])
    return len(meals_with)/numMeals * ingrWeight/dishWeight * 100

def calculate_wj(df, ingredient):
    """Calculate average weight of ingredient across all meals"""
    numMeals = len(df['Meal'].dropna())
    total = 0.0
    for idx in df[df['Ingredient'].str.lower()==ingredient.lower()].index:
        row_idx = idx
        while pd.isna(df.at[row_idx,'Meal']): row_idx -= 1
        dish = df.at[row_idx,'Meal']
        total += calculate_w_ij(df, dish, ingredient)
    return total/numMeals if numMeals else 0.0

def similarityF(food1, food2, cuisine_df):
    """Calculate similarity between two foods based on ingredients"""
    dish1_row = cuisine_df[cuisine_df['Meal'].str.lower() == food1.lower()]
    dish2_row = cuisine_df[cuisine_df['Meal'].str.lower() == food2.lower()]

    idx1 = dish1_row.index[0]
    count1 = int(dish1_row.iloc[0]['IngredientsCount'])
    ing1_df = cuisine_df.iloc[idx1:idx1 + count1][['Ingredient']].dropna()

    idx2 = dish2_row.index[0]
    count2 = int(dish2_row.iloc[0]['IngredientsCount'])
    ing2_df = cuisine_df.iloc[idx2:idx2 + count2][['Ingredient']].dropna()

    ing1_set = set(ing1_df['Ingredient'].str.lower())
    ing2_set = set(ing2_df['Ingredient'].str.lower())

    common_ingredients = ing1_set.intersection(ing2_set)

    total_similarity = 0.0
    for ingredient in common_ingredients:
        wj = calculate_wj(cuisine_df, ingredient)
        total_similarity += wj

    return total_similarity

def predictedContentBased(ratings_df, user_id, food, cuisine_df):
    """Predict rating using content-based filtering"""
    foodList = ratings_df['dish'].unique()
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    sumNum = 0
    sumDenom = 0

    for _, row in user_ratings.iterrows():
        current_meal = row['dish']
        current_rating = row['rating']

        if current_meal != food and current_rating != 0:
            sim = similarityF(food, current_meal, cuisine_df)
            sumNum += sim * current_rating
            sumDenom += sim

    if sumDenom == 0:
        return 0
    return sumNum / sumDenom

def average_rating(ratings_df, user_id):
    """Calculate average rating for a user"""
    sum_ratings = ratings_df.loc[ratings_df.user_id==user_id, 'rating'].sum()
    newRatings = ratings_df.loc[ratings_df.user_id==user_id, 'rating'][ratings_df.loc[ratings_df.user_id==user_id, 'rating'] != 0]
    if len(newRatings) == 0:
        return 0
    return float(sum_ratings / len(newRatings))

def months_since_oldest(Tp):
    """Calculate months since a timestamp"""
    today = datetime.today()
    return (today.year - Tp.year) * 12 + (today.month - Tp.month)

from datetime import date, datetime

def format_age(birth_date):
    today = date.today()
    days_old = (today - birth_date).days
    months = days_old // 30
    years = months // 12

    if days_old < 30:
        return f"{days_old} days ({years} years {months % 12} months)"
    elif months < 12:
        return f"{months} months ({years} years {months % 12} months)"
    else:
        return f"{years} years {months % 12} months"

def time_weight(ratings_df, u, v, meal, λ):
    """Calculate time-weighted similarity"""
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
    Tp = ratings_df['timestamp'].min()
    Tp = months_since_oldest(Tp)

    tu = ratings_df.loc[(ratings_df.user_id==u) & (ratings_df.dish==meal), 'timestamp'].iloc[0]
    tu = months_since_oldest(tu)

    tv = ratings_df.loc[(ratings_df.user_id==v) & (ratings_df.dish==meal), 'timestamp'].iloc[0]
    tv = months_since_oldest(tv)
    return math.sqrt(math.exp(-λ * (Tp - tu)) * math.exp(-λ * (Tp - tv)))

def time_weighted_user_similarity(ratings_df, u, v, λ):
    """Calculate time-weighted user similarity"""
    sum_numerator = 0
    sum_denominator = 0

    mean_u = average_rating(ratings_df, u)
    mean_v = average_rating(ratings_df, v)

    meals_u = set(ratings_df.loc[ratings_df.user_id==u, 'dish'])
    meals_v = set(ratings_df.loc[ratings_df.user_id==v, 'dish'])
    common = meals_u & meals_v
    if not common:
        return 0.0

    sumUDenominator = 0
    sumVDenominator = 0

    for meal in common:
        ru = ratings_df.loc[(ratings_df.user_id==u) & (ratings_df.dish==meal), 'rating'].iloc[0]
        rv = ratings_df.loc[(ratings_df.user_id==v) & (ratings_df.dish==meal), 'rating'].iloc[0]

        tw = time_weight(ratings_df, u, v, meal, λ)
        sum_numerator += (ru - mean_u) * (rv - mean_v) * tw

        sumUDenominator += (ru - mean_u) * (ru - mean_u) * tw
        sumVDenominator += (rv - mean_v) * (rv - mean_v) * tw

    sum_denominator = math.sqrt(sumUDenominator) * math.sqrt(sumVDenominator)

    return sum_numerator / sum_denominator if sum_denominator != 0 else 0

def predictedCollabRating(ratings_df, user_id, food, λ):
    """Predict rating using collaborative filtering"""
    sumNumerator = 0
    sumDenominator = 0

    mean_u = average_rating(ratings_df, user_id)

    for v in ratings_df.user_id.unique():
        if v != user_id:
            meals_v_row = ratings_df[ratings_df['dish'].str.lower() == food.lower()]
            if not meals_v_row.empty:
                meal_v_rating_row = meals_v_row.loc[ratings_df.user_id == v, 'rating']
                if not meal_v_rating_row.empty:
                    meal_v_rating = float(meal_v_rating_row.iloc[0])

                    sim = time_weighted_user_similarity(ratings_df, user_id, v, λ)

                    mean_v = average_rating(ratings_df, v)
                    sumNumerator += sim * (meal_v_rating - mean_v)
                    sumDenominator += abs(sim)

    if sumDenominator == 0:
        return 0
    return sumNumerator / sumDenominator + mean_u

def hybrid_filtering(food, user_id, beta, ratings_df, df, λ):
    """Combine content-based and collaborative filtering"""
    collab_rating = predictedCollabRating(ratings_df, user_id, food, λ)
    content_rating = predictedContentBased(ratings_df, user_id, food, df)

    if collab_rating == 0:
        return content_rating
    if content_rating == 0:
        return collab_rating
    return beta * collab_rating + (1 - beta) * content_rating

def health_safety_score(protein, PHE, energy, min_protein, min_PHE, max_PHE, min_energy, max_energy):
    """Calculate health safety score for a meal"""
    def calculate_distance(min_val, max_val, input_val):
        average = (min_val + max_val) / 2
        if input_val < min_val:
            return abs((average - input_val) / average * 100)
        elif input_val > max_val:
            return abs((average - input_val) / input_val * 100)
        else:
            return 0

    def safety_factor(distance):
        return 5 * (1 - distance / 100)

    protein_distance = 0
    if protein < min_protein:
        protein_distance = abs((min_protein - protein) / min_protein * 100)
    protein_safety_factor = 5 * (1 - protein_distance / 100)

    PHE_distance = calculate_distance(min_PHE, max_PHE, PHE)
    PHE_safety_factor = safety_factor(PHE_distance)

    energy_distance = calculate_distance(min_energy, max_energy, energy)
    energy_safety_factor = safety_factor(energy_distance)

    sf = (protein_safety_factor + PHE_safety_factor + energy_safety_factor) / 3

    return sf

def final_score(hybrid_score, hs_score, pref_scale):
    """Calculate final recommendation score"""
    return hybrid_score * (1 - pref_scale) + hs_score * pref_scale

def mealtime_nutrition(nutrient, mealtype):
    """Calculate nutrition needs for specific meal type"""
    if mealtype == "Lunch" or mealtype == "Dinner":
        return nutrient * 0.3
    else:  # Breakfast or Snack
        return nutrient * 0.2


# DISPLAY FUNCTIONS

def display_baby_diet_plan(result, weight_kg):
    """Display formatted baby diet plan"""
    
    st.markdown("---")
    st.header("📋 Daily Diet Plan")
    
    # Display nutritional targets
    needs = result['needs']
    st.subheader(f"Nutritional Targets ({needs['age_group']})")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Protein", f"{needs['protein_g']:.1f} g")
    with col2:
        st.metric("PHE Range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
    with col3:
        st.metric("Calories", f"{needs['energy_kcal']:.0f} kcal")
    
    # PHE Deletion Phase (if applicable)
    if result['deletion_needed']:
        st.markdown("---")
        st.error("⚠️ **PHASE 1: PHE DELETION PROTOCOL**")
        st.warning(f"Baby's PHE level ({result.get('current_phe_mg_dl', 0):.1f} mg/dL) requires immediate intervention")
        
        st.markdown(f"### Delete PHE for {result['deletion_hours']} hours")
        st.info(result['deletion_note'])
        
        deletion = result['deletion_phase']
        st.markdown("#### During Deletion Period (Phenex-1 ONLY):")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **{deletion['phenex']['phenex_type']}:**
            - Powder: **{deletion['phenex']['phenex_grams']:.1f} g** per 24 hours
            - Mix with water: **{deletion['water_ml']:.0f} mL**
            - Feedings per day: **{deletion['feedings_per_day']}**
            - Per feeding: **{deletion['water_ml'] / 7:.0f} mL** (approx)
            """)
        
        with col2:
            st.markdown(f"""
            **Provides:**
            - Protein: {deletion['phenex']['protein_g']:.1f} g
            - Calories: {deletion['phenex']['calories_kcal']:.0f} kcal
            - No PHE (PHE-free)
            """)
        
        st.error("🩸 **CRITICAL: Check blood PHE DAILY during deletion**")
        st.info("Stop deletion and begin Phase 2 when PHE drops to ≤5 mg/dL")
        
        st.markdown("---")
        st.success(f"**PHASE 2: ADDING PHE BACK** (After blood PHE ≤5 mg/dL)")
        st.info(f"Starting PHE dose: **{result['initial_phe_mg_per_kg']:.0f} mg/kg** = **{result['initial_phe_mg']:.0f} mg/day**")
    
    # Maintenance Diet (or Phase 2 after deletion)
    st.markdown("### Daily Formula + Phenex Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Formula (Similac With Iron)")
        st.markdown(f"""
        - **{result['formula']['formula_ml']:.0f} mL** per day
        - Provides:
          - PHE: {result['formula']['phe_mg']:.0f} mg
          - Protein: {result['formula']['protein_g']:.1f} g
          - Calories: {result['formula']['calories_kcal']:.0f} kcal
        - Divide into {result['feedings_per_day']} feedings
        - Per feeding: ~{result['formula']['formula_ml'] / 7:.0f} mL
        """)
    
    with col2:
        st.markdown(f"#### {result['phenex']['phenex_type']}")
        st.markdown(f"""
        - **{result['phenex']['phenex_grams']:.1f} g** powder per day
        - Provides:
          - Protein: {result['phenex']['protein_g']:.1f} g
          - Calories: {result['phenex']['calories_kcal']:.0f} kcal
          - No PHE
        """)
    
    st.markdown("#### Mixing Instructions")
    st.info(f"""
    1. Mix **{result['phenex']['phenex_grams']:.1f} g** {result['phenex']['phenex_type']} powder
    2. Add **{result['formula']['formula_ml']:.0f} mL** Similac With Iron
    3. Add **{result['water_ml']:.0f} mL** water
    4. **Total volume: {result['total_volume_ml']:.0f} mL** ({result['total_volume_ml']/29.574:.1f} fl oz)
    5. Divide into **{result['feedings_per_day']} feedings** per day
    6. Each feeding: approximately **{result['ml_per_feeding']:.0f} mL**
    """)
    
    # Daily Totals
    st.markdown("---")
    st.subheader("📊 Daily Nutrition Totals")
    
    totals = result['totals']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        protein_status = "✅" if abs(totals['protein_g'] - needs['protein_g']) < 1 else "⚠️"
        st.metric(
            "Total Protein",
            f"{totals['protein_g']:.1f} g",
            f"Target: {needs['protein_g']:.1f} g {protein_status}"
        )
    
    with col2:
        phe_in_range = needs['phe_mg_min'] <= totals['phe_mg'] <= needs['phe_mg_max']
        phe_status = "✅" if phe_in_range else "⚠️"
        st.metric(
            "Total PHE",
            f"{totals['phe_mg']:.0f} mg",
            f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg {phe_status}"
        )
    
    with col3:
        cal_status = "✅" if totals['calories_kcal'] >= needs['energy_kcal'] * 0.9 else "⚠️"
        st.metric(
            "Total Calories",
            f"{totals['calories_kcal']:.0f} kcal",
            f"Target: {needs['energy_kcal']:.0f} kcal {cal_status}"
        )
    
    # Monitoring Schedule
    st.markdown("---")
    st.subheader("📅 Monitoring Schedule")
    
    if result['deletion_needed']:
        st.warning("""
        **During Deletion Phase:**
        - Check blood PHE: **DAILY**
        - Stop deletion when PHE ≤5 mg/dL
        - If PHE drops below 2 mg/dL: Add formula immediately and contact clinic
        """)
    
    st.info("""
    **Ongoing Monitoring:**
    - Blood PHE checks: **Twice weekly** initially, then weekly
    - Target PHE range: **2-5 mg/dL** (120-300 µmol/L)
    - Weight: Measure **monthly**
    - Height: Measure **monthly**
    - Adjust PHE every 3 days if out of range
    """)
    
    with st.expander("📝 PHE Adjustment Guidelines"):
        st.markdown("""
        **If blood PHE is < 2 mg/dL:**
        - ADD 15 mg PHE (approximately 25 mL more formula)
        - Recheck in 3 days
        
        **If blood PHE is 5-10 mg/dL:**
        - REDUCE 15 mg PHE (approximately 25 mL less formula)
        - Recheck in 3 days
        
        **If blood PHE is > 10 mg/dL:**
        - REDUCE 30 mg PHE (approximately 50 mL less formula)
        - Recheck in 3 days
        - Contact metabolic clinic
        """)

def display_child_adult_plan(needs, age_months, weight_kg):
    """Display diet plan for children and adults"""
    
    st.markdown("---")
    st.header("📋 Daily Nutritional Targets")
    
    st.subheader(f"{needs['age_group']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Protein", f"{needs['protein_g']:.0f} g")
    with col2:
        st.metric("PHE Range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
    with col3:
        st.metric("Calories", f"{needs['energy_kcal']:.0f} kcal")
    
    st.markdown("---")
    st.subheader(f"🥫 Medical Food Requirement")
    
    # Calculate Phenex (assuming no protein from regular food initially)
    phenex = calculate_phenex_amount(needs['protein_g'], 0, age_months)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {phenex['phenex_type']}")
        st.markdown(f"""
        - **{phenex['phenex_grams']:.1f} g** powder per day
        - Divide into 4-6 servings
        - Per serving: ~{phenex['phenex_grams']/5:.1f} g
        """)
    
    with col2:
        st.markdown("#### Provides:")
        st.markdown(f"""
        - Protein: {phenex['protein_g']:.1f} g
        - Calories: {phenex['calories_kcal']:.0f} kcal
        - No PHE
        """)
    
    st.info(f"""
    **Mixing Instructions:**
    - Mix {phenex['phenex_grams']:.1f} g {phenex['phenex_type']} powder with water
    - Divide into 4-6 servings throughout the day
    - Chill to improve taste
    - Can be mixed with allowed fruits or flavoring
    """)
    
    st.markdown("---")
    st.subheader("🍎 Regular Food Allowance")
    
    st.markdown(f"""
    You can have **{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg PHE** from regular foods per day.
    
    Use the **Custom Meal Planner** tab to:
    - Select foods from low-PHE options
    - Calculate PHE content of your meals
    - Plan your daily menu within your PHE target
    """)
    
    # Monitoring
    st.markdown("---")
    st.subheader("📅 Monitoring Schedule")
    st.info("""
    - Blood PHE checks: **Weekly** 
    - Target PHE range: **2-5 mg/dL** for children, **2-10 mg/dL** for adults
    - Weight: Check **monthly**
    - Adjust diet if PHE is consistently out of range
    """)

# STREAMLIT APP

def main():
    st.set_page_config(page_title="PKU Diet Manager", layout="wide")
    
    # Initialize session state
    if 'profile_created' not in st.session_state:
        st.session_state.profile_created = False
    
    # Landing Page
    if not st.session_state.profile_created:
        st.title("PKU Diet Management System")
        st.markdown("""
        ### Welcome to your personalized PKU diet planner!
        
        This application helps manage phenylketonuria (PKU) diet by calculating:
        - Daily phenylalanine (PHE) targets
        - Phenex medical food requirements  
        - Formula amounts for infants
        - Meal planning for children and adults
        
        **Let's start by creating your profile.**
        """)
        
        st.markdown("---")
        st.header("Create Profile")
        
        # Age group selection
        age_category = st.radio(
            "Who is this profile for?",
            ["Baby (0-12 months)", "Child (1-12 years)", "Adult (12+ years)"]
        )
        
        st.markdown("---")
        
        # Common fields
        col1, col2 = st.columns(2)
        
        with col1:
            units = st.radio("Units:", ["Metric", "Imperial"])
            
            if units == "Metric":
                weight = st.number_input('Weight (kg):', min_value=0.0, step=0.1, key='weight')
                height_cm = st.number_input('Height (cm):', min_value=0.0, step=1.0, key='height_cm')
                height = height_cm / 100  # Convert to meters
                st.session_state['user_height_cm'] = height_cm  # Store height in cm
                st.session_state['user_height'] = height  # Store height in meters
            else:
                weight_lbs = st.number_input('Weight (lbs):', min_value=0.0, step=0.1, key='weight_lbs')
                weight = weight_lbs * 0.453592
                height_in = st.number_input('Height (inches):', min_value=0.0, step=1.0, key='height_in')
                height = height_in * 0.0254
        
        with col2:
            st.markdown("**Birth Date:**")
            birth_year = st.number_input('Year:', min_value=1900, max_value=datetime.now().year, value=2025, key='birth_year')
            birth_month = st.number_input('Month:', min_value=1, max_value=12, value=1, key='birth_month')
            birth_day = st.number_input('Day:', min_value=1, max_value=31, value=1, key='birth_day')
            
            current_phe = st.number_input(
                'Current Blood PHE Level (mg/dL):', 
                min_value=0.0, 
                step=0.1,
                help="Enter your most recent blood phenylalanine level",
                key='current_phe'
            )
        
        # Age-specific fields
        sex = None
        age_hours = None
        
        if age_category != "Baby (0-12 months)":
            sex = st.radio("Biological Sex:", ["Male", "Female"])
        else:
            # For babies ≤96 hours, ask age in hours
            age_months = calculate_age_months(birth_year, birth_month, birth_day)
            if age_months:# Less than 1 month old
                age_hours = st.number_input(
                    'Baby\'s age (hours):', 
                    min_value=0, 
                    max_value=720,
                    help="Enter age in hours (only needed if baby is 96 hours old or less)",
                    key='age_hours'
                )
        
        st.markdown("---")
        
        if st.button("Calculate Diet Plan", type="primary"):
            if weight == 0 or height == 0:
                st.error("Please enter valid weight and height")
        else:
        # Save to session state
            st.session_state.profile_created = True
            st.session_state.user_age_category = age_category  
            st.session_state.user_weight = weight
            st.session_state.user_height_cm = height_cm
            st.session_state.user_height = height
            st.session_state.user_birth_month = birth_month
            st.session_state.user_birth_year = birth_year
            st.session_state.user_birth_day = birth_day
            st.session_state.user_current_phe = current_phe
            st.session_state.user_sex = sex
            st.session_state.user_age_hours = age_hours if age_category == "Baby (0-12 months)" else None
            st.rerun()
    
    # Main Application (after profile created)
    else:
        # Sidebar with profile summary
        st.sidebar.header("👤 Profile")
        age_months = calculate_age_months(
            st.session_state.birth_year,
            st.session_state.birth_month, 
            st.session_state.birth_day
        )
        
        st.sidebar.write(f"**Age:** {age_months} months ({age_months//12} years {age_months%12} months)")
        st.sidebar.write(f"**Weight:** {st.session_state.weight:.1f} kg")
        st.sidebar.write(f"**Height:** {st.session_state['user_height_cm']:.1f} cm")
        st.sidebar.write(f"**Current PHE:** {st.session_state.current_phe:.1f} mg/dL")
        

        if 'user_birth_year' in st.session_state and 'user_birth_month' in st.session_state and 'user_birth_day' in st.session_state:
            birth_date = date(
            int(st.session_state['user_birth_year']),
            int(st.session_state['user_birth_month']),
            int(st.session_state['user_birth_day'])
    )
        else:
            st.error("Birth date information is missing. Please create a profile first.")
            st.stop()

        age_display = format_age(birth_date)
        st.write(f"👤 Profile")
        st.write(f"Age: {age_display}")

        if st.sidebar.button("🔄 Create New Profile"):
            st.session_state.profile_created = False
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Main content area
        st.title("PKU Diet Plan")
        
        # Calculate and display appropriate diet plan
        if st.session_state.age_category == "Baby (0-12 months)":
            result = calculate_baby_diet(
                age_months,
                st.session_state.weight,
                st.session_state.current_phe,
                st.session_state.age_hours
            )
            result['current_phe_mg_dl'] = st.session_state.current_phe
            display_baby_diet_plan(result, st.session_state.weight)
            
        else:
            # Child or Adult
            needs = get_child_adult_daily_needs(
                age_months,
                st.session_state.weight,
                st.session_state.sex
            )
            display_child_adult_plan(needs, age_months, st.session_state.weight)
        
        # Additional tabs for meal planning, etc.
        st.markdown("---")
        tabs = st.tabs(["Custom Meal Planner", "Food Database", "Resources"])
        
        with tabs[0]:
            st.header("🍽️ Custom Meal Planner")
            
            if st.session_state.age_category == "Baby (0-12 months)":
                if age_months < 6:
                    st.info("Babies under 6 months should only have formula and Phenex as calculated above. Complementary foods will be introduced after 6 months.")
                else:
                    st.markdown("""
                    ### Adding Solid Foods (6-12 months)
                    
                    Your baby can now start eating solid foods! Use the calculator below to see how solid foods 
                    affect the formula amount needed.
                    """)
                    
                    # Load fruit/vegetable data if available
                    try:
                        # This would use your existing CSV files
                        st.info("Feature coming soon: Select baby foods and automatically adjust formula amount")
                    except:
                        st.warning("Food database not loaded")
            
            else:
                st.markdown("""
                ### Plan Your Meals Within Your PHE Target
                
                Use this tool to select foods and track your daily PHE intake.
                """)
                
                # Get user's PHE target
                if st.session_state.age_category == "Child (1-12 years)":
                    needs = get_child_adult_daily_needs(age_months, st.session_state.weight, st.session_state.sex)
                else:
                    needs = get_child_adult_daily_needs(age_months, st.session_state.weight, st.session_state.sex)
                
                st.metric("Your Daily PHE Target", f"{needs['phe_mg_min']:.0f} - {needs['phe_mg_max']:.0f} mg")
                
                # Meal tracker
                if 'meal_tracker' not in st.session_state:
                    st.session_state.meal_tracker = []
                
                st.subheader("Add Foods to Your Day")
                
                # Sample food additions (you would expand this with your CSV data)
                food_categories = {
                    "Fruits": [
                        ("Apple (100g)", 5, 0.2, 59),
                        ("Banana (100g)", 16, 0.4, 89),
                        ("Orange (100g)", 14, 0.4, 47),
                        ("Grapes (100g)", 16, 0.8, 86)
                    ],
                    "Vegetables": [
                        ("Carrots (100g)", 15, 0.4, 18),
                        ("Lettuce (100g)", 14, 0.3, 4),
                        ("Tomato (100g)", 14, 0.5, 12),
                        ("Cucumber (100g)", 16, 0.6, 14)
                    ],
                    "Grains (Limited)": [
                        ("Rice (30g)", 32, 0.6, 33),
                        ("Pasta (30g)", 33, 0.6, 21)
                    ]
                }
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    category = st.selectbox("Food Category:", list(food_categories.keys()))
                    food_options = food_categories[category]
                    food_names = [f[0] for f in food_options]
                    selected_food = st.selectbox("Select Food:", food_names)
                    servings = st.number_input("Servings:", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
                
                with col2:
                    if st.button("➕ Add to Day"):
                        # Find the food data
                        food_data = next(f for f in food_options if f[0] == selected_food)
                        st.session_state.meal_tracker.append({
                            'food': food_data[0],
                            'servings': servings,
                            'phe_mg': food_data[1] * servings,
                            'protein_g': food_data[2] * servings,
                            'calories': food_data[3] * servings
                        })
                        st.rerun()
                
                # Display meal tracker
                if st.session_state.meal_tracker:
                    st.markdown("---")
                    st.subheader("Today's Foods")
                    
                    total_phe = 0
                    total_protein = 0
                    total_calories = 0
                    
                    for idx, item in enumerate(st.session_state.meal_tracker):
                        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                        with col1:
                            st.write(f"{item['food']}")
                        with col2:
                            st.write(f"×{item['servings']:.1f}")
                        with col3:
                            st.write(f"{item['phe_mg']:.0f} mg")
                        with col4:
                            st.write(f"{item['protein_g']:.1f}g")
                        with col5:
                            if st.button("🗑️", key=f"delete_{idx}"):
                                st.session_state.meal_tracker.pop(idx)
                                st.rerun()
                        
                        total_phe += item['phe_mg']
                        total_protein += item['protein_g']
                        total_calories += item['calories']
                    
                    st.markdown("---")
                    st.subheader("Daily Totals")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        phe_status = "✅" if needs['phe_mg_min'] <= total_phe <= needs['phe_mg_max'] else "⚠️"
                        st.metric(
                            "PHE from Foods",
                            f"{total_phe:.0f} mg {phe_status}",
                            f"Target: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg"
                        )
                    
                    with col2:
                        # Calculate remaining protein needed from Phenex
                        protein_from_phenex = needs['protein_g'] - total_protein
                        phenex = calculate_phenex_amount(needs['protein_g'], total_protein, age_months)
                        st.metric(
                            f"{phenex['phenex_type']} Needed",
                            f"{phenex['phenex_grams']:.1f} g"
                        )
                    
                    with col3:
                        total_cal_with_phenex = total_calories + phenex['calories_kcal']
                        st.metric(
                            "Total Calories",
                            f"{total_cal_with_phenex:.0f} kcal",
                            f"Target: {needs['energy_kcal']:.0f} kcal"
                        )
                    
                    # Visual representation
                    if total_phe > 0:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            name='Current PHE',
                            x=['PHE Intake'],
                            y=[total_phe],
                            marker_color='lightblue'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            name='Target Range',
                            x=['PHE Intake', 'PHE Intake'],
                            y=[needs['phe_mg_min'], needs['phe_mg_max']],
                            mode='lines',
                            line=dict(color='green', width=3, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="PHE Intake vs Target Range",
                            yaxis_title="PHE (mg)",
                            showlegend=True,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("🗑️ Clear All Foods"):
                        st.session_state.meal_tracker = []
                        st.rerun()
        
        with tabs[1]:
            st.header("📚 Food Database")
            st.markdown("""
            ### Low-PHE Food Options
            
            Below are some examples of low-phenylalanine foods. Always check portion sizes!
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Very Low PHE (< 15 mg per serving)")
                st.markdown("""
                **Fruits:**
                - Apples, pears, grapes
                - Oranges, grapefruit
                - Berries (strawberries, blueberries)
                - Watermelon, cantaloupe
                
                **Vegetables:**
                - Lettuce, cucumber
                - Tomatoes, peppers
                - Carrots, celery
                - Green beans, squash
                """)
            
            with col2:
                st.subheader("Low PHE (15-30 mg per serving)")
                st.markdown("""
                **Grains (small portions):**
                - Rice (30g cooked)
                - Pasta (30g cooked)
                - Corn (small amounts)
                
                **Other:**
                - Special low-protein breads
                - Special low-protein pasta
                - Some cereals (limited)
                """)
            
            st.warning("""
            ⚠️ **HIGH PHE FOODS TO AVOID:**
            - Meat, poultry, fish
            - Eggs and dairy products
            - Nuts and seeds
            - Regular bread and pasta (large amounts)
            - Beans and lentils
            - Soy products
            """)
        
        with tabs[2]:
            st.header("📖 PKU Management Resources")
            
            st.subheader("Understanding Your Numbers")
            st.markdown("""
            **Blood PHE Levels:**
            - **Target for children:** 2-5 mg/dL (120-300 µmol/L)
            - **Target for adults:** 2-10 mg/dL (120-600 µmol/L)
            - Levels should be checked regularly (weekly to monthly depending on age and control)
            
            **What affects PHE levels:**
            - Amount of PHE eaten from foods
            - Adequate calorie and protein intake
            - Illness or stress
            - Growth spurts in children
            """)
            
            st.subheader("Tips for Success")
            st.markdown("""
            **For Babies:**
            1. Mix Phenex fresh daily and refrigerate
            2. Warm bottles to room temperature (never microwave)
            3. Keep accurate records of all feedings
            4. Watch for signs of PHE deficiency: poor weight gain, lethargy, rash
            5. Never skip blood tests - they guide your baby's diet
            
            **For Children & Adults:**
            1. Take Phenex medical food consistently (divide throughout the day)
            2. Measure foods with a food scale for accuracy
            3. Plan meals in advance
            4. Keep emergency low-PHE snacks available
            5. Communicate with your metabolic team regularly
            """)
            
            st.subheader("When to Contact Your Metabolic Clinic")
            st.error("""
            **Contact immediately if:**
            - Blood PHE > 15 mg/dL
            - Blood PHE < 1 mg/dL
            - Baby is refusing feedings
            - Unexpected weight loss
            - Persistent vomiting or diarrhea
            - Unusual lethargy or behavior changes
            """)
            
            st.info("""
            **Routine Contact:**
            - Before making major diet changes
            - When starting new medications
            - If having trouble following the diet
            - For recipe ideas and support
            - When planning pregnancy (females)
            """)
            
            st.subheader("Important Reminders")
            st.warning("""
            - This app is a tool to help manage your PKU diet
            - Always follow your metabolic team's specific recommendations
            - Never make major diet changes without consulting your doctor/dietitian
            - Blood PHE monitoring is essential - never skip tests
            - PKU is manageable with proper diet and monitoring
            """)

# RUN APPLICATION

if __name__ == "__main__":
    main()