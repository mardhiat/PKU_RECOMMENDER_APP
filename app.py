import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
from datetime import datetime
import math
import os
import plotly.graph_objects as go
import plotly.express as px
import io

# ========== NEW: PHE CAP CONFIGURATION ==========
MAX_PHE_ADULT = 750  # mg/day for adults (>12 months)
MAX_PHE_CHILD = None  # Will be calculated based on age/weight

def apply_phe_cap(meal_plan, total_phe, max_phe, age_in_months):
    """
    Applies PHE cap by scaling portions proportionally if needed.
    Returns adjusted meal plan and warning message if applicable.
    """
    if age_in_months < 12:  # Baby - use guideline ranges
        return meal_plan, None
    
    # For children and adults
    if total_phe > max_phe:
        scale_factor = max_phe / total_phe
        warning_msg = f"⚠️ Total daily PHE ({total_phe:.0f} mg) exceeds recommended limit ({max_phe:.0f} mg). Portions have been adjusted by {scale_factor:.1%} to stay within safe limits."
        
        # Scale all portions
        adjusted_plan = {}
        for meal, details in meal_plan.items():
            adjusted_plan[meal] = {
                'portion': details['portion'] * scale_factor,
                'phe': details['phe'] * scale_factor,
                'protein': details['protein'] * scale_factor,
                'calories': details['calories'] * scale_factor
            }
        
        return adjusted_plan, warning_msg
    
    return meal_plan, None

def get_phe_limit(age_in_months, weight):
    """
    Returns the PHE limit based on age and clinical practice.
    For adults (>12 months), uses 750 mg/day cap.
    """
    if age_in_months >= 12:
        return MAX_PHE_ADULT
    else:
        # For babies, use calculated need
        return calcNeedOfPhe(age_in_months, weight)
# ================================================


def calcAge(year, month, day):
    today = datetime.today()
    return (today.year - year) * 12 + (today.month - month)

def calculate_bmi(weight, height):
    if height == 0:
        return 0
    bmi = weight / (height * height)
    return bmi

def timeNeeded(weight, cals):
    results = []
    for met in (2.5, 3.3, 4.0):
        total_min = cals / (met * weight) * 60
        hrs = math.floor(total_min / 60)
        mins = round(total_min - hrs * 60)
        results.append((hrs, mins))
    return results

def calcNeedOfProtein(weight, age):
    if age < 3: return 3.25 * weight
    if age < 6: return 3.25 * weight
    if age < 9: return 2.75 * weight
    return 2.75 * weight

def calcNeedOfCals(weight, age):
    if age < 3: return 120 * weight
    if age < 6: return 120 * weight
    if age < 9: return 107.5 * weight
    return 107.5 * weight

def calcNeedOfPhe(age, weight):
    """Calculate PHE needs based on age in months and weight"""
    if age < 6:  # 0-6 months
        return 47.5 * weight
    elif age < 12:  # 6-12 months
        return 32.5 * weight
    elif age < 144:  # 1-12 years (12-144 months)
        return 30 * weight
    else:  # 12 years and above
        return 22.5 * weight

def prepare_meal_df(raw_df: pd.DataFrame) -> pd.DataFrame:
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
    """Load all cuisine files and create a dictionary"""
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
            st.warning(f"File {filename} not found. Please ensure all CSV files are in the app directory.")
    
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

def calculate_w_ij(df, dish, ingredient):
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
    numMeals = len(df['Meal'].dropna())
    total = 0.0
    for idx in df[df['Ingredient'].str.lower()==ingredient.lower()].index:
        row_idx = idx
        while pd.isna(df.at[row_idx,'Meal']): row_idx -= 1
        dish = df.at[row_idx,'Meal']
        total += calculate_w_ij(df, dish, ingredient)
    return total/numMeals if numMeals else 0.0

def similarityF(food1, food2, cuisine_df):
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
    sum = ratings_df.loc[ratings_df.user_id==user_id, 'rating'].sum()

    newRatings = ratings_df.loc[ratings_df.user_id==user_id, 'rating'][ratings_df.loc[ratings_df.user_id==user_id, 'rating'] != 0]
    if len(newRatings) == 0:
        return 0
    return float(sum / len(newRatings))

def months_since_oldest(Tp):
    today = datetime.today()
    return (today.year - Tp.year) * 12 + (today.month - Tp.month)

def time_weight(ratings_df, u, v, meal, λ):
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
    Tp = ratings_df['timestamp'].min()
    Tp = months_since_oldest(Tp)

    tu = ratings_df.loc[(ratings_df.user_id==u) & (ratings_df.dish==meal), 'timestamp'].iloc[0]
    tu = months_since_oldest(tu)

    tv = ratings_df.loc[(ratings_df.user_id==v) & (ratings_df.dish==meal), 'timestamp'].iloc[0]
    tv = months_since_oldest(tv)
    return math.sqrt(math.exp(-λ * (Tp - tu)) * math.exp(-λ * (Tp - tv)))

def user_similarity(ratings_df, u, v, λ):
    meals_u = set(ratings_df.loc[ratings_df.user_id==u, 'dish'])
    meals_v = set(ratings_df.loc[ratings_df.user_id==v, 'dish'])
    common = meals_u & meals_v
    if not common:
        return 0.0
    mean_u = ratings_df.loc[ratings_df.user_id==u, 'rating'].mean()
    mean_v = ratings_df.loc[ratings_df.user_id==v, 'rating'].mean()
    num = denom_u = denom_v = 0.0
    for meal in common:
        ru = ratings_df.loc[(ratings_df.user_id==u) & (ratings_df.dish==meal), 'rating'].iloc[0]
        rv = ratings_df.loc[(ratings_df.user_id==v) & (ratings_df.dish==meal), 'rating'].iloc[0]
        w = time_weight(ratings_df, u, v, meal, λ)
        num += w * (ru - mean_u) * (rv - mean_v)
        denom_u += w * (ru - mean_u)**2
        denom_v += w * (rv - mean_v)**2
    return num / (math.sqrt(denom_u) * math.sqrt(denom_v)) if denom_u and denom_v else 0.0

def predict_rating(ratings_df, u, meal, λ):
    mean_u = ratings_df.loc[ratings_df.user_id==u, 'rating'].mean()
    sims = []
    devs = []
    for v in ratings_df.user_id.unique():
        if v == u:
            continue
        rvj = ratings_df.loc[(ratings_df.user_id==v) & (ratings_df.dish==meal), 'rating']
        if rvj.empty:
            continue
        s = user_similarity(ratings_df, u, v, λ)
        sims.append(s)
        devs.append(s * (rvj.iloc[0] - ratings_df.loc[ratings_df.user_id==v, 'rating'].mean()))
    return mean_u + sum(devs) / sum(abs(np.array(sims))) if sims else mean_u

def time_weighted_user_similarity(ratings_df, u, v, λ):
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

    return sum_numerator / sum_denominator

def predictedCollabRating(ratings_df, user_id, food, λ):
    sumNumerator = 0
    sumDenominator = 0

    mean_u = average_rating(ratings_df, user_id)

    for v in ratings_df.user_id.unique():
        if v != user_id:
            meals_v_row = ratings_df[ratings_df['dish'].str.lower() == food.lower()]
            meal_v_rating = float(meals_v_row.loc[ratings_df.user_id == v, 'rating'].iloc[0])

            sim = time_weighted_user_similarity(ratings_df, user_id, v, λ)

            mean_v = average_rating(ratings_df, v)
            sumNumerator += sim * (meal_v_rating - mean_v)
            sumDenominator += sim

    if sumDenominator == 0:
        return 0
    return sumNumerator / sumDenominator + mean_u

def hybrid_filtering(food, user_id, beta, ratings_df, df, λ):
    collab_rating = predictedCollabRating(ratings_df, user_id, food, λ)
    content_rating = predictedContentBased(ratings_df, user_id, food, df)

    if collab_rating == 0:
        return content_rating
    if content_rating == 0:
        return collab_rating
    return beta * collab_rating + (1 - beta) * content_rating

def calculate_distance(min, max, input):
    average = (min + max) / 2
    if input < min:
        return abs((average - input) / average * 100)
    elif input > max:
        return abs((average - input) / input * 100)
    else:
        return 0

def safety_factor(distance):
    return 5 * (1 - distance / 100)

def health_safety_score(protein, PHE, energy, min_protein, min_PHE, max_PHE, min_energy, max_energy):
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
    return hybrid_score * (1 - pref_scale) + hs_score * pref_scale

def meal_nutrition(dish, serving_size, cuisine_file, ingredients_file):
    dish_row = cuisine_file[cuisine_file['Meal'].str.lower() == dish.lower()]
    if dish_row.empty:
        raise ValueError(f"Dish '{dish}' not found in cuisine file")
    
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

def mealtime_nutrition(nutrient, mealtime):
    if mealtime == "Lunch/Dinner":
        return nutrient * .3
    if mealtime == "Breakfast/Snack":
        return nutrient * .2

RATINGS_FILE = 'ratings.csv'

def append_ratings_file(df):
    header = not os.path.exists(RATINGS_FILE)
    df.to_csv(RATINGS_FILE, mode='a', index=False, header=header)

# Initialize session state variables
if 'ratings_session' not in st.session_state:
    st.session_state.ratings_session = pd.DataFrame(
        columns=['user_id','cuisine','meal_type','dish','rating','timestamp']
    )

if 'protein' not in st.session_state:
    st.session_state['protein'] = 0
    st.session_state['phe1'] = 0
    st.session_state['phe2'] = 0
    st.session_state['e1'] = 0
    st.session_state['e2'] = 0
    st.session_state['phe'] = 0
    st.session_state['energy'] = 0
    st.session_state['age_in_months'] = 0  # NEW: Track age
    st.session_state['weight'] = 0  # NEW: Track weight

# Load data files
cuisine_data = load_cuisine_files()
ing_df, chat_df = load_ingredient_files()

# Sidebar configuration
st.sidebar.header('Settings')
user_id = st.sidebar.number_input('User ID', 1, 10, key='user_id')
beta = st.sidebar.slider('Recommendation Style (People - 1 vs. Ingredients - 0)', 0.0, 1.0, 0.5, key='beta')
λ = st.sidebar.slider('Preference Recency (0 = All Ratings Equal; 5 = Prioritize Recent Rating)', 0.0, 5.0, 0.0, key='λ')
pref_scale = st.sidebar.slider('Preference - Health Safety (0 most preference, 1 most healthy)', 0.0, 1.0, 0.5, key='Preference Scale')

# Main app header
st.title("PKU Diet Management System")
st.markdown("""
### Welcome to your personalized PKU diet planner!

This app helps you manage your phenylketonuria (PKU) diet by providing:
- **Baby Diet Planning** for infants under 12 months
- **Personalized Profile** to calculate your nutritional needs
- **Food Ratings** to rate dishes from different cuisines
- **Smart Recommendations** based on your preferences and health requirements
- **AI Chat Assistant** for PKU nutrition questions
- **Custom Meal Planning** with portion calculations
""")

# Available cuisines info
st.sidebar.markdown("### Available Cuisines")
st.sidebar.markdown("""
- African Foods
- Central European Foods  
- Chinese Foods
- Eastern European Foods
- Indian Foods
- Italian Foods
- Japanese Foods
- Mediterranean Foods
- Mexican Foods
- Scottish-Irish Foods
""")

tabs = st.tabs(["Baby Diet Planner", "Profile", "Ratings", "Recommendation", "Chat", "Custom Meal Planner"])

with tabs[0]:
    st.header("Baby Diet Planner (0-12 months)")
    st.markdown("""
    ### Instructions:
    1. Enter your baby's current PHE level, weight, height, and birth date
    2. Select your preferred unit system (Metric/Imperial)
    3. Click 'Calculate Diet' to get nutritional requirements
    4. Choose between milk/formula feeding or complementary food planning
    """)
    
    if 'diet_calculated' not in st.session_state:
        st.session_state.diet_calculated = False
    phe = st.number_input('PHE level (mg):', 0.0)
    wtype = st.selectbox('Units:', ['', 'Metric','Imperial'], key='diet_wt')

    if wtype != '':
        if wtype == 'Metric':
            weight = st.number_input('Weight (kg):', 0.0, key='diet_weight')
            height = st.number_input('Height (cm):', 0.0, key='diet_height') / 100
        if wtype == 'Imperial':
            weight = st.number_input('Weight (lbs):', 0.0, key='diet_weight') / 2.205
            height = st.number_input('Height (inches):', 0.0, key='diet_height') / 39.37
        
        year = st.number_input('Birth year:', 2023, datetime.now().year, key='diet_year')
        month = st.number_input('Birth month:', 1, 12, key='diet_month')
        day = st.number_input('Birth day:', 1, 31, key='diet_day')

        if st.button('Calculate Diet', key='calc_diet'):
            st.session_state.diet_calculated = True

        if st.session_state.diet_calculated:
            age = calcAge(year, month, day)
            if height != 0:
                bmi = weight/height**2
            else:
                bmi = 0

            needP = calcNeedOfProtein(weight, age)
            needC = calcNeedOfCals(weight, age)
            needPh = calcNeedOfPhe(age, weight)
            st.write(f'Age: {age} months, BMI: {bmi:.1f}')
            st.write(f'Daily Requirements - Protein: {needP:.1f}g, Calories: {needC:.0f}, PHE: {needPh:.0f}mg')

with tabs[1]:
    st.header('Personal Profile & Nutrition Calculator')
    st.markdown("""
    ### Instructions:
    1. Select your preferred unit system
    2. Choose your biological sex
    3. Enter your current weight, height, and birth date
    4. Click 'Calculate Nutrition' to get your daily nutritional targets
    """)

    units = st.radio("Units:", ["Metric", "Imperial"], index=0)
    sex = st.radio("Sex:", ["Male", "Female"], index=0)

    if(units == "Metric"):
        weight = st.number_input('Weight (kg): ', min_value=0.0, step=1.0)
        height = st.number_input('Height (m): ', min_value=0.0, step=.01)
    elif(units == "Imperial"):
        weight = st.number_input('Weight (lbs): ', min_value=0.0, step=1.0)
        height = st.number_input('Height (inches): ', min_value=0.0, step=.01)
        weight = weight * 0.453592
        height = height * 0.0254

    bmi = calculate_bmi(weight, height)
    bmi_category = ""
    if bmi < 18.5:
            bmi_category = "underweight"
    elif bmi < 25:
            bmi_category = "normal"
    elif bmi < 30:
            bmi_category = "overweight"
    elif bmi < 35:
            bmi_category = "obesity - class I"
    elif bmi < 40:
            bmi_category = "obesity - class II"
    else:
            bmi_category = "obesity - class III"

    year = st.number_input('Birth year:', 1900, datetime.now().year, key='diet1_year')
    month = st.number_input('Birth month:', 1, 12, key='diet1_month')
    day = st.number_input('Birth day:', 1, 31, key='diet1_day')
    age_in_months = calcAge(year, month, day)

    if st.button('Calculate Nutrition', key='nutr'):
        e1 = 0
        e2 = 0
        protein = 0
        phe1 = 0
        phe2 = 0
        e3 = 0

        def mean(n1, n2):
            return (float(n1) + n2) / 2

        if(age_in_months < 12):
            st.write("Please refer to baby diet planner instead")
        elif(age_in_months < 132):
            if(age_in_months < 48):
                e1 = 900
                e2 = 1800
                e3 = 1300
                phe1 = 4  # Lower bound
                phe2 = 8  # Upper bound (for avg calculation)
                protein = 30
        elif(age_in_months < 84):
                e1 = 1300
                e2 = 2300
                e3 = 1700
                phe1 = 4  # Lower bound
                phe2 = 8  # Upper bound
                protein = 35
        else:
                e1 = 1650
                e2 = 3300
                e3 = 2400
                phe1 = 8  # Lower bound
                phe2 = 16  # Upper bound
                protein = 40
    
        if(bmi_category == "underweight"):
            energy = e2
        elif(bmi_category == "normal"):
            energy = e3
        else:
            energy = e1
    else:
        if(sex == "Female"):
            if(age_in_months < 180):
                e1 = 1500
                e2 = 3000
                e3 = 2200
                phe1 = 12  # Lower bound
                phe2 = 24  # Upper bound
                protein = 50
        elif(age_in_months < 228):
                e1 = 1200
                e2 = 3000
                e3 = 2100
                phe1 = 12  # Lower bound
                phe2 = 24  # Upper bound
                protein = 55
        else:
                e1 = 1400
                e2 = 2500
                e3 = 2100
                phe1 = 12  # Lower bound
                phe2 = 24  # Upper bound
                protein = 60
    else:  # Male
        if(age_in_months < 180):
                e1 = 2000
                e2 = 3700
                e3 = 2700
                phe1 = 12  # Lower bound
                phe2 = 24  # Upper bound
                protein = 55
        elif(age_in_months < 228):
                e1 = 2100
                e2 = 3900
                e3 = 2800
                phe1 = 12  # Lower bound
                phe2 = 24  # Upper bound
                protein = 65
        else:
                e1 = 2000
                e2 = 3300
                e3 = 2900
                phe1 = 12  # Lower bound
                phe2 = 24  # Upper bound
                protein = 70
    
        if(bmi_category == "underweight"):
            energy = e2
        elif(bmi_category == "normal"):
            energy = e3
        else:
            energy = e1

# Calculate average PHE as (phe1 + phe2) / 2
    phe = (phe1 + phe2) / 2
        
        # ========== NEW: Apply 750mg cap for adults ==========
        if age_in_months >= 12:
            actual_phe_limit = MAX_PHE_ADULT
            if phe > actual_phe_limit:
                st.warning(f"⚠️ Clinical Practice Note: While guidelines allow {phe1}-{phe2} mg/day, we limit adult PHE intake to {actual_phe_limit} mg/day to prevent elevated phenylalanine levels.")
                phe = actual_phe_limit
        # ====================================================

        st.write(f"**Your Daily Nutritional Targets:**")
        st.write(f"- Daily Calorie Goal: {energy} kcal")
        st.write(f"- Daily Protein Goal: {protein} g")
        
        # ========== NEW: Show clinical limit for adults ==========
        if age_in_months >= 12:
            st.write(f"- Daily Phenylalanine Limit (Clinical): **{MAX_PHE_ADULT} mg** (guideline range: {phe1}-{phe2} mg)")
        else:
            st.write(f"- Daily Phenylalanine Range: {phe1}-{phe2} mg (Average: {phe:.0f} mg)")
        # ========================================================
        
        st.write(f"- BMI: {bmi:.1f} ({bmi_category})")

        # Store in session state
        st.session_state['protein'] = protein
        st.session_state['phe1'] = phe1
        st.session_state['phe2'] = phe2
        st.session_state['e1'] = e1
        st.session_state['e2'] = e2
        st.session_state['phe'] = phe
        st.session_state['energy'] = energy
        st.session_state['age_in_months'] = age_in_months  # NEW
        st.session_state['weight'] = weight  # NEW
        
        st.success("Profile saved! You can now get personalized recommendations.")

with tabs[2]:
    st.header('Rate Cuisine Dishes')
    st.markdown("""
    ### Instructions:
    1. Select which cuisine you'd like to try from the dropdown below
    2. Choose a meal type (breakfast, lunch, dinner, etc.)
    3. Rate dishes on a scale of 0-5 (0 = haven't tried, 5 = love it)
    4. Click 'Submit Ratings' to save your preferences
    """)
    
    if not cuisine_data:
        st.warning("No cuisine data available. Please ensure CSV files are in the app directory.")
    else:
        cuisine_choice = st.selectbox('Select Cuisine to Rate:', list(cuisine_data.keys()))
        
        if cuisine_choice:
            cuisine_df = cuisine_data[cuisine_choice]
            meal_types = cuisine_df['MealType'].unique().tolist()
            meal_type = st.selectbox('Meal Type:', meal_types, key='rating_meal_type')
            
            df_filtered = cuisine_df[cuisine_df['MealType']==meal_type]
            dishes = df_filtered['MealGroup'].unique().tolist()
            
            if dishes:
                st.markdown(f"**Rate dishes from {cuisine_choice} - {meal_type}:**")
                ratings = {}
                for dish in dishes:
                    ratings[dish] = st.slider(f"{dish}", 0, 5, 0, key=f'r_{dish}')
                
                if st.button('Submit Ratings', key='submit_ratings'):
                    records = []
                    for dish, rating in ratings.items():
                        records.append({
                            'user_id': user_id,
                            'cuisine': cuisine_choice,
                            'meal_type': meal_type,
                            'dish': dish,
                            'rating': rating,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    new_df = pd.DataFrame(records)
                    st.session_state.ratings_session = pd.concat([st.session_state.ratings_session, new_df], ignore_index=True)
                    st.success('Ratings saved successfully!')

with tabs[3]:
    st.header('Personalized Food Recommendations')
    st.markdown("""
    ### Instructions:
    1. Make sure you've completed your profile and rated some dishes
    2. Enter your desired serving size in grams
    3. Choose between getting top 5 recommendations or analyzing a specific dish
    4. View nutritional information and safety scores for recommended foods
    """)
    
    serving_size = st.number_input('Serving Size (grams):', min_value=1.0, value=100.0, step=10.0)
    
    if not cuisine_data or ing_df is None:
        st.warning("Nutritional database not available. Please ensure all data files are loaded.")
    elif st.session_state.ratings_session.empty:
        st.info("Please rate some dishes in the 'Ratings' tab first to get personalized recommendations.")
    else:
        rtabs = st.tabs(['Top 5 Recommendations', 'Single Food Recommendation'])
        meal_category = "Lunch/Dinner"
        protein = st.session_state['protein']
        phe1 = st.session_state['phe1']
        phe2 = st.session_state['phe2']
        e1 = st.session_state['e1']
        e2 = st.session_state['e2']
        age_in_months = st.session_state.get('age_in_months', 0)

        proteinM = mealtime_nutrition(protein, meal_category)
        phe1M = mealtime_nutrition(phe1, meal_category)
        phe2M = mealtime_nutrition(phe2, meal_category)
        e1M = mealtime_nutrition(e1, meal_category)
        e2M = mealtime_nutrition(e2, meal_category)
        
        # ========== NEW: Use clinical PHE limit ==========
        if age_in_months >= 12:
            phe_limit = MAX_PHE_ADULT
        else:
            phe_limit = st.session_state['phe']
        # ================================================
        
        with rtabs[0]:
            st.subheader('Top 5 Recommendations')
            
            # ========== NEW: Show PHE limit being used ==========
            if age_in_months >= 12:
                st.info(f"🔒 Daily PHE limit: {MAX_PHE_ADULT} mg (clinical practice)")
            # ==================================================
            
            user_ratings = st.session_state.ratings_session[st.session_state.ratings_session['user_id'] == user_id]
            if not user_ratings.empty:
                rated_cuisine = user_ratings['cuisine'].iloc[0]
                
                if rated_cuisine in cuisine_data:
                    cuisine_df = cuisine_data[rated_cuisine]
                    
                    if st.button('Get Top 5 Meal Recommendations', key='get_recommendations'):
                        all_meals = cuisine_df['Meal'].dropna().unique()
                        rated_meals = user_ratings[user_ratings['rating'] > 0]['dish'].unique()
                        unrated_meals = [meal for meal in all_meals if meal not in rated_meals]

                        predictions = []
                        total_daily_phe = 0  # NEW: Track total PHE
                        
                        for meal in unrated_meals:
                            try:
                                vector = meal_nutrition(meal, serving_size, cuisine_df, ing_df)
                                meal_protein = vector[0]
                                meal_phenyl = vector[1] 
                                meal_calories = vector[2]
                                
                                hyb_score = hybrid_filtering(meal, user_id, beta, st.session_state.ratings_session, cuisine_df, λ)
                                hesa_score = health_safety_score(meal_protein, meal_phenyl, meal_calories, proteinM, phe1M, phe2M, e1M, e2M)
                                fi_score = final_score(hyb_score, hesa_score, pref_scale)
                                predictions.append((meal, fi_score, meal_protein, meal_phenyl, meal_calories))
                            except Exception as e:
                                continue

                        if predictions:
                            top5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
                            
                            # ========== NEW: Calculate total PHE and apply cap ==========
                            total_daily_phe = sum(item[3] for item in top5)
                            
                            if age_in_months >= 12 and total_daily_phe > MAX_PHE_ADULT:
                                scale_factor = MAX_PHE_ADULT / total_daily_phe
                                st.warning(f"⚠️ Total PHE ({total_daily_phe:.0f} mg) exceeds {MAX_PHE_ADULT} mg limit. Portions adjusted by {scale_factor:.1%}.")
                                
                                # Scale all portions
                                top5 = [(dish, score, protein * scale_factor, phe * scale_factor, calories * scale_factor) 
                                       for dish, score, protein, phe, calories in top5]
                                total_daily_phe = MAX_PHE_ADULT
                            # ===========================================================
                            
                            st.subheader("Top 5 Recommended Dishes")
                            st.write(f"**Total Daily PHE: {total_daily_phe:.0f} mg / {phe_limit:.0f} mg**")
                            
                            for dish, score, protein, phe, calories in top5:
                                st.write(f"**{dish}**: Score {score:.2f}")
                                st.write(f"   - PHE: {phe:.1f}mg, Protein: {protein:.1f}g, Calories: {calories:.1f}kcal")
                            
                            data = {
                                'Meal': [item[0] for item in top5],
                                'Phenylalanine': [item[3] for item in top5],
                                'Protein': [item[2] for item in top5], 
                                'Energy': [item[4] for item in top5]
                            }
                            
                            df_viz = pd.DataFrame(data)
                            fig = px.bar(df_viz, x="Meal", y=["Protein", "Phenylalanine"], 
                                       barmode="group", title="Nutritional Content of Top Recommendations")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No recommendations available. Try rating more dishes first.")
                else:
                    st.error(f"Cuisine data for '{rated_cuisine}' not found.")
            else:
                st.info("No ratings found. Please rate some dishes first.")
        
        with rtabs[1]:
            st.subheader("Single Food Analysis")
            
            analysis_cuisine = st.selectbox("Select cuisine for analysis:", list(cuisine_data.keys()))
            
            if analysis_cuisine:
                cuisine_df = cuisine_data[analysis_cuisine]
                available_meals = cuisine_df['Meal'].dropna().unique().tolist()
                meal = st.selectbox('Choose meal for detailed analysis:', available_meals)
                
                if meal:
                    try:
                        cscore = predictedContentBased(st.session_state.ratings_session, user_id, meal, cuisine_df)
                        colscore = predictedCollabRating(st.session_state.ratings_session, user_id, meal, λ)
                        hscore = hybrid_filtering(meal, user_id, beta, st.session_state.ratings_session, cuisine_df, λ)

                        vector = meal_nutrition(meal, serving_size, cuisine_df, ing_df)
                        meal_protein = vector[0]
                        meal_phenyl = vector[1]
                        meal_calories = vector[2]

                        hs_score = health_safety_score(meal_protein, meal_phenyl, meal_calories, proteinM, phe1M, phe2M, e1M, e2M)
                        f_score = final_score(hscore, hs_score, pref_scale)
                        
                        st.write(f'**Scores for {meal}:**')
                        st.write(f'- Content Score: {cscore:.2f}')
                        st.write(f'- Collaborative Score: {colscore:.2f}')
                        st.write(f'- Hybrid Score: {hscore:.2f}')
                        st.write(f'- Health Safety Score: {hs_score:.2f}')
                        st.write(f'- **Final Score: {f_score:.2f}**')
                        
                        # ========== NEW: PHE limit check ==========
                        if age_in_months >= 12 and meal_phenyl > MAX_PHE_ADULT:
                            st.error(f"⚠️ This serving ({meal_phenyl:.0f} mg PHE) exceeds daily limit of {MAX_PHE_ADULT} mg!")
                            safe_portion = (MAX_PHE_ADULT / meal_phenyl) * serving_size
                            st.info(f"💡 Recommended safe portion: {safe_portion:.0f}g")
                        # =========================================

                        if st.session_state['phe'] > 0:
                            labels = ['Phenylalanine', 'Protein', 'Energy']
                            
                            percent_phe = (meal_phenyl / phe_limit) * 100
                            percent_protein = (meal_protein / st.session_state['protein']) * 100 if st.session_state['protein'] > 0 else 0
                            percent_calories = (meal_calories / st.session_state['energy']) * 100 if st.session_state['energy'] > 0 else 0
                            
                            meal_values = [percent_phe, percent_protein, percent_calories]
                            user_limits = [100, 100, 100]

                            fig = go.Figure()
                            fig.add_trace(go.Scatterpolar(
                                r=user_limits,
                                theta=labels,
                                fill='toself',
                                name='Daily Limit (100%)',
                                opacity=0.3
                            ))
                            fig.add_trace(go.Scatterpolar(
                                r=meal_values,
                                theta=labels,
                                fill='toself',
                                name=f'{meal} (per {serving_size}g)'
                            ))

                            fig.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, max(max(meal_values), 100)])),
                                showlegend=True,
                                title="Meal Nutrition vs Daily Limits"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Could not analyze {meal}: {str(e)}")
                        st.info("Make sure the dish exists in the selected cuisine database.")

with tabs[4]:
    st.header('PKU AI Assistant')
    st.markdown("""
    ### Instructions:
    Ask me anything about PKU nutrition! I can help with:
    - Food choices and phenylalanine content
    - Meal planning suggestions  
    - Nutritional information
    - PKU management tips
    """)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**PKU Assistant:** {message['content']}")
    
    with st.form("chat_form", clear_on_submit=True):
        chat_input = st.text_input("Ask your PKU nutrition question:")
        send_clicked = st.form_submit_button("Send Message")
    
    if send_clicked and chat_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": chat_input})
        
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            context = f"""You are a helpful PKU nutrition assistant. The user asked: {chat_input}
            
            Provide helpful, accurate information about PKU diet management, low-phenylalanine foods, 
            and general nutrition advice for people with phenylketonuria."""
            
            response = model.generate_content(context)
            st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            st.rerun()
            
        except Exception as e:
            st.error(f"Chat service temporarily unavailable: {str(e)}")

with tabs[5]:
    st.header("Custom Meal Planner")
    st.markdown("""
    ### Instructions:
    1. Select your meal type (breakfast, lunch, dinner, snack)
    2. Choose your desired cuisine from our available options
    3. Enter your serving size preferences
    4. View calculated PHE content and recommended portions
    5. Get personalized meal suggestions based on your daily limits
    """)
    
    if not cuisine_data or ing_df is None:
        st.warning("Nutritional database not available. Please ensure all data files are loaded.")
    else:
        meal_type_selection = st.selectbox("Select Meal Type:", 
                                         ["Breakfast", "Lunch", "Dinner", "Snack"])
        
        cuisine_selection = st.selectbox("Select Cuisine:", list(cuisine_data.keys()))
        
        if cuisine_selection and meal_type_selection:
            cuisine_df = cuisine_data[cuisine_selection]
            
            available_dishes = cuisine_df['Meal'].dropna().unique().tolist()
            st.subheader(f"Available {cuisine_selection} dishes:")
            st.write(f"Total dishes available: {len(available_dishes)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Quick Portion Calculator")
                selected_dish = st.selectbox("Choose a dish:", available_dishes)
                serving_size_custom = st.number_input("Serving size (grams):", min_value=1.0, value=100.0)
                
                if selected_dish:
                    try:
                        nutrition_vector = meal_nutrition(selected_dish, serving_size_custom, cuisine_df, ing_df)
                        protein, phe, calories = nutrition_vector[0], nutrition_vector[1], nutrition_vector[2]
                        
                        st.write(f"**Nutritional content per {serving_size_custom}g:**")
                        st.write(f"- Phenylalanine: {phe:.1f} mg")
                        st.write(f"- Protein: {protein:.1f} g")
                        st.write(f"- Calories: {calories:.1f} kcal")
                        
                        # ========== NEW: Use clinical PHE limit ==========
                        age_in_months = st.session_state.get('age_in_months', 0)
                        user_daily_phe = MAX_PHE_ADULT if age_in_months >= 12 else st.session_state.get('phe', 0)
                        # ===============================================
                        
                        if user_daily_phe > 0:
                            meal_phe_ratio = 0.3 if meal_type_selection in ["Lunch", "Dinner"] else 0.2
                            meal_phe_limit = user_daily_phe * meal_phe_ratio
                            
                            if phe > 0:
                                safe_portion = (meal_phe_limit / phe) * serving_size_custom
                                st.info(f"**Recommended safe portion:** {safe_portion:.0f}g for {meal_type_selection.lower()}")
                                
                                # ========== NEW: Warn if exceeds limit ==========
                                if phe > meal_phe_limit:
                                    st.warning(f"⚠️ Current serving exceeds meal PHE limit ({meal_phe_limit:.0f} mg)")
                                # ==============================================
                        
                    except Exception as e:
                        st.error(f"Could not calculate nutrition for {selected_dish}. Please check ingredient database.")
            
            with col2:
                st.subheader("Dish Browser")
                dish_search = st.text_input("Search dishes:")
                
                if dish_search:
                    filtered_dishes = [dish for dish in available_dishes 
                                     if dish_search.lower() in dish.lower()]
                    st.write(f"Found {len(filtered_dishes)} matching dishes:")
                    for dish in filtered_dishes[:10]:
                        st.write(f"• {dish}")
                else:
                    st.write("Sample dishes from this cuisine:")
                    for dish in available_dishes[:10]:
                        st.write(f"• {dish}")
            
            st.subheader("Compare Multiple Dishes")
            dishes_to_compare = st.multiselect("Select dishes to compare (max 5):", 
                                             available_dishes, max_selections=5)
            
            if dishes_to_compare and len(dishes_to_compare) > 1:
                comparison_data = []
                for dish in dishes_to_compare:
                    try:
                        nutrition = meal_nutrition(dish, 100, cuisine_df, ing_df)
                        comparison_data.append({
                            'Dish': dish,
                            'PHE (mg/100g)': round(nutrition[1], 1),
                            'Protein (g/100g)': round(nutrition[0], 1),
                            'Calories (kcal/100g)': round(nutrition[2], 1)
                        })
                    except:
                        continue
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    fig = px.bar(comparison_df, x='Dish', y=['PHE (mg/100g)', 'Protein (g/100g)'], 
                               barmode='group', title='Nutritional Comparison (per 100g)')
                    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
### About the Data
This app uses comprehensive nutritional databases covering:
- **10 International Cuisines** with traditional dishes and recipes
- **Detailed Ingredient Database** with phenylalanine, protein, and calorie content
- **Baby Food Database** for infants 6-12 months
- **AI-Powered Chat** for personalized PKU nutrition guidance

**Clinical Note:** For adults and children over 12 months, this app enforces a **750 mg/day PHE limit** based on clinical practice, 
even though guidelines may allow higher amounts. This helps prevent elevated phenylalanine levels.

*Always consult with your healthcare provider or registered dietitian before making significant changes to your PKU diet.*
""")