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
    if age < 3: return 47.5 * weight
    if age < 6: return 32.5 * weight
    if age < 9: return 30 * weight
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

def similarityF(food1, food2):
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

def predictedContentBased(ratings_df, user_id, food):
    # Get all meals
    foodList = ratings_df['dish'].unique()

    # Get ratings for this user
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    sumNum = 0
    sumDenom = 0

    for _, row in user_ratings.iterrows():
        current_meal = row['dish']
        current_rating = row['rating']

        if current_meal != food and current_rating != 0:
            sim = similarityF(food, current_meal)
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

def time_weight(ratings_df, u, v, meal):
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
    Tp = ratings_df['timestamp'].min()
    Tp = months_since_oldest(Tp)

    tu = ratings_df.loc[(ratings_df.user_id==u) & (ratings_df.dish==meal), 'timestamp'].iloc[0]
    tu = months_since_oldest(tu)

    tv = ratings_df.loc[(ratings_df.user_id==v) & (ratings_df.dish==meal), 'timestamp'].iloc[0]
    tv = months_since_oldest(tv)
    return math.sqrt(math.exp(-λ * (Tp - tu)) * math.exp(-λ * (Tp - tv)))

def user_similarity(ratings_df, u, v):
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
        w = time_weight(ratings_df, u, v, meal)
        num += w * (ru - mean_u) * (rv - mean_v)
        denom_u += w * (ru - mean_u)**2
        denom_v += w * (rv - mean_v)**2
    return num / (math.sqrt(denom_u) * math.sqrt(denom_v)) if denom_u and denom_v else 0.0

def predict_rating(ratings_df, u, meal):
    mean_u = ratings_df.loc[ratings_df.user_id==u, 'rating'].mean()
    sims = []
    devs = []
    for v in ratings_df.user_id.unique():
        if v == u:
            continue
        rvj = ratings_df.loc[(ratings_df.user_id==v) & (ratings_df.dish==meal), 'rating']
        if rvj.empty:
            continue
        s = user_similarity(ratings_df, u, v)
        sims.append(s)
        devs.append(s * (rvj.iloc[0] - ratings_df.loc[ratings_df.user_id==v, 'rating'].mean()))
    return mean_u + sum(devs) / sum(abs(np.array(sims))) if sims else mean_u

def time_weighted_user_similarity(ratings_df, u, v):
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

        tw = time_weight(ratings_df, u, v, meal)
        sum_numerator += (ru - mean_u) * (rv - mean_v) * tw

        sumUDenominator += (ru - mean_u) * (ru - mean_u) * tw
        sumVDenominator += (rv - mean_v) * (rv - mean_v) * tw

    sum_denominator = math.sqrt(sumUDenominator) * math.sqrt(sumVDenominator)

    return sum_numerator / sum_denominator

def predictedCollabRating(ratings_df, user_id, food):
    sumNumerator = 0
    sumDenominator = 0

    mean_u = average_rating(ratings_df, user_id)

    for v in ratings_df.user_id.unique():
        if v != user_id:
            meals_v_row = ratings_df[ratings_df['dish'].str.lower() == food.lower()]
            meal_v_rating = float(meals_v_row.loc[ratings_df.user_id == v, 'rating'].iloc[0])

            sim = time_weighted_user_similarity(ratings_df, user_id, v)

            mean_v = average_rating(ratings_df, v)
            sumNumerator += sim * (meal_v_rating - mean_v)
            sumDenominator += sim

    if sumDenominator == 0:
        return 0
    return sumNumerator / sumDenominator + mean_u

def hybrid_filtering(food, user_id, beta, ratings_df, df):
    collab_rating = predictedCollabRating(ratings_df, user_id, food)
    content_rating = predictedContentBased(ratings_df, user_id, food)

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
    numIngredients = int(dish_row.iloc[0]['Number of Ingredients'])

    dishProtein = 0
    dishPhenyl = 0
    dishCalories = 0

    for i in range(dish_row.index[0], dish_row.index[0] + numIngredients):
        ingredient = cuisine_file.iloc[i]['Ingredient']
        ingredient_row = ingredients_file[ingredients_file['Ingredient'].str.lower() == ingredient.lower()]

        grams = float(ingredient_row.iloc[0, 1])
        gramsWanted = float(cuisine_file.iloc[i, 3])
        gramsOfProtein = gramsWanted / grams * float(ingredient_row.iloc[0, 4])
        gramsOfPhenyl = gramsWanted / grams * float(ingredient_row.iloc[0, 2])
        calories = gramsWanted / grams * float(ingredient_row.iloc[0, 5])

        dishProtein += gramsOfProtein
        dishPhenyl += gramsOfPhenyl
        dishCalories += calories

    mealGrams = dish_row.iloc[0, 1]

    conversion = serving_size / mealGrams
    dishProtein *= conversion
    dishPhenyl *= conversion
    dishCalories *= conversion

    vector = [dishProtein, dishPhenyl, dishCalories]

    return vector

def mealtime_nutrition(nutrient, mealtime):
    if mealtime == "Lunch/Dinner":
        return nutrient * .3
    if mealtime == "Breakfast/Snack":
        return nutrient * .2

RATINGS_FILE = 'ratings.csv'

def append_ratings_file(df):
    header = not os.path.exists(RATINGS_FILE)
    df.to_csv(RATINGS_FILE, mode='a', index=False, header=header)

st.sidebar.header('File Uploads & Settings')
baby6 = st.sidebar.file_uploader('Baby Food 6-8 mo CSV', type='csv', key='baby6')
baby8 = st.sidebar.file_uploader('Baby Food 8-12 mo CSV', type='csv', key='baby8')

cuisine_files = st.sidebar.file_uploader('Cuisine Meal CSVs', type='csv', accept_multiple_files=True, key='cuisines')
choice = None
if cuisine_files:
    names = [f.name for f in cuisine_files]
    choice = st.sidebar.selectbox('Select Cuisine', names, key='cuisine_choice')
    raw = next(f for f in cuisine_files if f.name == choice)
    cuisine_df = prepare_meal_df(pd.read_csv(raw, encoding='latin1'))
else:
    cuisine_df = None
ing_file = st.sidebar.file_uploader('Extra Ingredient CSV', type='csv', key='ingredient')

chat_ing = st.sidebar.file_uploader('Chat Ingredient CSV', type='csv', key='chat_ing')
user_id = st.sidebar.number_input('User ID',1,10,key='user_id')
beta = st.sidebar.slider('Hybrid β',0.0,1.0,0.5,key='beta')
λ = st.sidebar.slider('λ Time Weight', 0.0, 5.0, 0.0, key='λ')
pref_scale = st.sidebar.slider('Preference - Health Safety (0 most preference, 1 most healthy)', 0.0, 1.0, 0.5, key='Preference Scale')

ing_df = pd.read_csv(ing_file) if ing_file else None
chat_df = pd.read_csv(chat_ing) if chat_ing else None
df8 = pd.read_csv(baby8) if baby8 else None

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

tabs = st.tabs(["Baby Diet Planner", "Profile", "Ratings", "Recommendation", "Chat", "Custom Meal Planner"])
with tabs[0]:
    if 'diet_calculated' not in st.session_state:
        st.session_state.diet_calculated = False
    phe = st.number_input('PHE level (mg):', 0.0)
    wtype = st.selectbox('Units:', ['', 'Metric','Imperial'], key='diet_wt')

    if wtype != '':
        if wtype == 'Metric':
            weight = st.number_input('Weight (kg):', 0.0, key='diet_weight')
        if wtype == 'Metric':
            height = st.number_input('Height (cm):', 0.0, key='diet_height') / 100

        if wtype == 'Imperial':
            weight = st.number_input('Weight (lbs):', 0.0, key='diet_weight') / 2.205
        if wtype == 'Imperial':
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
            st.write(f'Age: {age} mo, BMI: {bmi:.1f}')
            st.write(f'Protein: {needP:.1f}g, Calories: {needC:.0f}, PHE: {needPh:.0f}')
    
    tabD = st.tabs(["Milk and Formula Diet (<6 Months)", "Complementary Food (>6 Months)"])
    with tabD[0]:
        st.header('PHE Diet & Baby Food Calculator')
        if st.session_state.diet_calculated:
            if baby6 and baby8:
                feed = st.selectbox('Feed with:', ['','Breast Milk','Formula'], index = 0)
                if feed=='Breast Milk':
                    amt=needPh*100/48
                    prot=amt*1.07/100
                    phenex = max((needP-prot)*100/15,0)
                    st.write(f'{amt:.0f}g {feed} + {phenex:.0f}g Phenex')
                if feed=='Formula':
                    amt=needPh*100/395
                    prot=amt*9.7/100
                    phenex = max((needP-prot)*100/15,0)
                    st.write(f'{amt:.0f}g {feed} + {phenex:.0f}g Phenex')
            else:
                st.info('Upload Baby Food CSVs in the sidebar')
    with tabD[1]:
        st.header("Custom Daily Meal Planner")

        if not baby8:
            st.warning("Please upload the Baby Food CSV file.")
            st.stop()

        try:
            baby8.seek(0)
            df8 = pd.read_csv(baby8)
            df8.columns = df8.columns.str.strip().str.lower().str.replace(" ", "")
            st.subheader("Baby Food File Preview")
            st.dataframe(df8.head())
            required_cols = ["ingredient", "phe(mg)", "protein(g)", "energy(kcal)", "servingsize(g)"]
            if df8.empty or df8.columns.size == 0:
                st.error("\u26a0\ufe0f Ingredient CSV file is empty or malformed.")
                st.stop()
            if not all(col in df8.columns for col in required_cols):
                st.error("\u26a0\ufe0f Ingredient CSV missing required columns: " + ", ".join([col for col in required_cols if col not in ing_df.columns]))
                st.stop()
        except Exception as e:
            st.error(f"\u26a0\ufe0f Failed to read ingredient file: {e}")
            st.stop()
        if st.session_state.diet_calculated:
            milk = st.radio("Milk Type", ["Breast Milk", "Formula"])
            need_p = needP
            need_ph = needPh
            need_c = needC
            uFoodRange = st.slider("% of PHE from food", 10, 100, 50)
            num_foods = st.number_input("Number of foods", 1, 10, step=1)
            portion_type = st.radio("Portion style", ["Equal", "Custom"])

            all_ing = df8['ingredient'].dropna().unique().tolist() if 'ingredient' in df8.columns else []
            if not all_ing:
                st.warning("No valid ingredients found in the uploaded file.")
                st.stop()

            foods = [st.selectbox(f"Food #{i+1}", all_ing, key=f"f{i}") for i in range(num_foods)]
            portions = [st.slider(f"% {f}", 0, 100, 10, key=f"p{i}") if portion_type == "Custom" else uFoodRange / num_foods for i, f in enumerate(foods)]
            st.text(f"Debug PHE: need_ph={need_ph}, uFoodRange={uFoodRange}")

            def calc(food, portion):
                row = df8[df8['ingredient'].str.lower() == food.lower()]
                if row.empty: return 0, 0, 0
                phe = float(row['phe(mg)'].values[0])
                weight = float(row['servingsize(g)'].values[0])
                grams = (need_ph * uFoodRange * portion) / (phe * 10000)
                prot = grams * float(row['protein(g)'].values[0]) / weight
                cal = grams * float(row['energy(kcal)'].values[0]) / weight
                return grams, prot, cal

            if st.button("Calculate Plan"):
                total_p, total_c = 0, 0
                grams_list = []
                food_summary = []
                for f, pct in zip(foods, portions):
                    g, p, c = calc(f, pct)
                    grams_list.append((f, g))
                    total_p += p
                    total_c += c
                    food_summary.append({"Food": f, "Grams": g, "Protein": p, "Calories": c})

                milk_p = (1.07 if milk == "Breast Milk" else 1.54)
                milk_cal = (70 if milk == "Breast Milk" else 68)
                phe_per_100 = (48 if milk == "Breast Milk" else 67.8)
                milk_g = (100 - uFoodRange) * need_ph / phe_per_100
                milk_total_p = milk_g * milk_p / 100
                milk_total_c = milk_g * milk_cal / 100
                phenex_g = max(0.0, need_p - total_p - milk_total_p) * 100 / 15
                phenex_cal = phenex_g * 480 / 100

                st.subheader("Daily Meal Summary")
                st.write(f"Milk: {milk_g:.1f}g {milk}, {milk_total_c:.0f} kcal")
                st.write(f"Phenex: {phenex_g:.1f}g, {phenex_cal:.0f} kcal")
                for f, g in grams_list:
                    st.write(f"{f}: {g:.1f}g")
                total_kcal = total_c + milk_total_c + phenex_cal
                st.write(f"Total Calories: {total_kcal:.0f} kcal")

                # Show chart
                df_chart = pd.DataFrame(food_summary)
                df_chart.loc[len(df_chart)] = {"Food": "Milk", "Grams": milk_g, "Protein": milk_total_p, "Calories": milk_total_c}
                df_chart.loc[len(df_chart)] = {"Food": "Phenex", "Grams": phenex_g, "Protein": need_p - total_p - milk_total_p, "Calories": phenex_cal}
                fig = px.bar(df_chart, x="Food", y=["Protein", "Calories"], barmode="group")
                st.plotly_chart(fig)

                # Export CSV
                if st.download_button("Download Meal Plan as CSV", df_chart.to_csv(index=False), file_name="meal_plan.csv"):
                    st.success("CSV download triggered")
            else:
                st.info("Upload Cuisine and Ingredient CSVs first")

    #-------------------------------------------- End of Bella Tab 5 :')
with tabs[1]:
    st.header('Profile')

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
    if height != 0:
        bmi = weight/height**2
    else:
        bmi = 0
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

    def mean(n1, n2):
        return (float(n1) + n2) / 2

    year = st.number_input('Birth year:', 1900, datetime.now().year, key='diet1_year')
    month = st.number_input('Birth month:', 1, 12, key='diet1_month')
    day = st.number_input('Birth day:', 1, 31, key='diet1_day')
    age_in_months = calcAge(year, month, day)
    st.write(age_in_months)

    if st.button('Calculate Nutrition', key='nutr'):
        e1 = 0
        e2 = 0
        protein = 0
        phe1 = 0
        phe2 = 0
        e3 = 0

        if(age_in_months < 12):
            st.write("Please refer to baby diet planner instead")
        elif(age_in_months < 132):
            if(age_in_months < 48):
                e1 = 900
                e2 = 1800
                e3 = (e1+e2)/2
                phe1 = 200
                phe2 = 400
                protein = 30
            elif(age_in_months < 84):
                e1 = 1300
                e2 = 2300
                e3 = 1700
                phe1 = 210
                phe2 = 450
                protein = 35
            else:
                e1 = 1650
                e2 = 3300
                e3 = 2400
                phe1 = 220
                phe2 = 500
                protein = 40
            if(bmi_category == "underweight"):
                energy = e2
            if(bmi_category == "normal"):
                energy = e3
            if(bmi_category == "overweight" or bmi_category == "obesity - class I" or bmi_category == "obesity - class II" or bmi_category == "obesity - class III"):
                energy = e1
        else:
            if(sex == "Female"):
                if(age_in_months < 180):
                    e1 = 1500
                    e2 = 3000
                    e3 = 2200
                    phe1 = 250
                    phe2 = 700
                    protein = 50
                elif(age_in_months < 228):
                    e1 = 1200
                    e2 = 3000
                    e3 = 2100
                    phe1 = 230
                    phe2 = 700
                    protein = 55
                else:
                    e1 = 1400
                    e2 = 2500
                    e3 = 2100
                    phe1 = 220
                    phe2 = 700
                    protein = 60
            if(sex == "Male"):
                if(age_in_months < 180):
                    e1 = 2000
                    e2 = 3700
                    e3 = 2700
                    phe1 = 225
                    phe2 = 900
                    protein = 55
                elif(age_in_months < 228):
                    e1 = 2100
                    e2 = 3900
                    e3 = 2800
                    phe1 = 295
                    phe2 = 1100
                    protein = 65
                else:
                    e1 = 2000
                    e2 = 3300
                    e3 = 2900
                    phe1 = 290
                    phe2 = 1200
                    protein = 70
            if(bmi_category == "underweight"):
                energy = e2
            if(bmi_category == "normal"):
                energy = e3
            if(bmi_category == "overweight" or bmi_category == "obesity - class I" or bmi_category == "obesity - class II" or bmi_category == "obesity - class III"):
                energy = e1
            phe = mean(phe1, phe2)

            st.write(f"Daily Calorie Goal: {energy}")
            st.write(f"Daily Protein Goal: {protein}")
            st.write(f"Daily Phenylalanine Goal: {phe}")


            st.session_state['protein'] = protein
            st.session_state['phe1'] = phe1
            st.session_state['phe2'] = phe2
            st.session_state['e1'] = e1
            st.session_state['e2'] = e2
            st.session_state['phe'] = phe
            st.session_state['energy'] = energy
            st.write(f"Daily Phenylalanine Goal: {st.session_state['phe']}")


with tabs[2]:
    st.header('Cuisine Ratings')
    if cuisine_df is not None:
        meal_types = cuisine_df['MealType'].unique().tolist()
        meal_type = st.selectbox('Meal Type', meal_types, key='rating_meal_type')
        df_code = cuisine_df[cuisine_df['MealType']==meal_type]
        dishes = df_code['MealGroup'].unique().tolist()
        if dishes:
            ratings = {d: st.slider(d,0,5,0,key=f'r_{d}') for d in dishes}
            if st.button('Submit Ratings', key='submit_ratings'):
                recs=[]
                for d,r in ratings.items():
                    recs.append({
                        'user_id':user_id,
                        'cuisine':choice,
                        'meal_type':meal_type,
                        'dish':d,
                        'rating':r,
                        'timestamp':datetime.now().isoformat()
                    })
                newdf = pd.DataFrame(recs)
                st.session_state.ratings_session = pd.concat([st.session_state.ratings_session,newdf],ignore_index=True)
                append_ratings_file(newdf)
                st.success('Ratings saved')
        else:
            st.warning('No dishes available for this meal type.')
        st.subheader('Session Ratings')
        st.dataframe(st.session_state.ratings_session)
        st.subheader('Permanent Ratings')
        if os.path.exists(RATINGS_FILE): st.dataframe(pd.read_csv(RATINGS_FILE))
        else: st.info('No permanent ratings file')
    else:
        st.info('Upload Cuisine CSVs in the sidebar')

with tabs[3]:
    serving_size = st.number_input('Serving Size (g): ', min_value=0.0, step=10.0)
    rtabs = st.tabs(['Top 5 Recommendations', 'Single Food Recommendation'])
    meal_category = "Lunch/Dinner"
    protein = st.session_state['protein']
    phe1 = st.session_state['phe1']
    phe2 = st.session_state['phe2']
    e1 = st.session_state['e1']
    e2 = st.session_state['e2']

    proteinM = mealtime_nutrition(protein, meal_category)
    phe1M = mealtime_nutrition(phe1, meal_category)
    phe2M = mealtime_nutrition(phe2, meal_category)
    e1M = mealtime_nutrition(e1, meal_category)
    e2M = mealtime_nutrition(e2, meal_category)
    with rtabs[0]:
        st.header('Recommendations')
        if cuisine_df is not None and not st.session_state.ratings_session.empty:
            if st.button('Recommend Top 5 Meals', key='Meals'):
                all_meals = cuisine_df['Meal'].dropna().unique()

                rated_meals = st.session_state.ratings_session[
                    (st.session_state.ratings_session['user_id'] == user_id) &
                    (st.session_state.ratings_session['rating'] > 0)
                ]['dish'].unique()

                unrated_meals = [meal for meal in all_meals if meal not in rated_meals]

                predictions = []
                for meal in unrated_meals:
                    try:
                        vector = meal_nutrition(meal, serving_size, cuisine_df, ing_df)
                        meal_protein = vector[0]
                        meal_phenyl = vector[1]
                        meal_calories = vector[2]
                        hyb_score = hybrid_filtering(meal, user_id, beta, st.session_state.ratings_session, cuisine_df)
                        hesa_score = health_safety_score(meal_protein, meal_phenyl, meal_calories, proteinM, phe1M, phe2M, e1M, e2M)
                        fi_score = final_score(hyb_score, hesa_score, pref_scale)
                        predictions.append((meal, fi_score))
                    except Exception as e:
                        h = "h"
                top5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

                st.subheader("Top 5 Recommended Dishes")
                for dish, score in top5:
                    st.write(f"**{dish}**: {score:.2f}")
                # for dish, score in predictions:
                #     st.write(f"**{dish}**: {score:.2f}")


                data = {
                    'Meal': [],
                    'Phenylalanine': [],
                    'Protein': [],
                    'Energy': []
                }

                for dish, score in top5:
                    vec = meal_nutrition(dish, serving_size, cuisine_df, ing_df)
                    protein = vec[0]
                    phenyl = vec[1]
                    energy = vec[2]

                    data['Meal'].append(dish)
                    data['Phenylalanine'].append(phenyl)
                    data['Protein'].append(protein)
                    data['Energy'].append(energy)

                df = pd.DataFrame(data)

                df.set_index('Meal').plot(kind='bar', stacked=True, figsize=(8,5))
                plt.title('Nutritional Content per Food')
                plt.ylabel('Amount (mg / g / kcal')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)

                fig = px.scatter(df, x="Protein", y="Phenylalanine",
                                 size="Energy", color="Meal",
                                 hover_name="Meal", size_max=60,
                                 title="Protein vs. Phenylalanine (Bubble size = Energy)")
                st.plotly_chart(fig)
        else:
            st.info('Upload cuisine CSVs and rate some dishes first')
    with rtabs[1]:
        st.header("Single Food Recommendation")
        meal = st.text_input('Meal for recommendation:', key='rec_meal')
        if meal:
            cscore = predictedContentBased(st.session_state.ratings_session, user_id, meal)
            colscore = predictedCollabRating(st.session_state.ratings_session, user_id, meal)
            hscore = hybrid_filtering(meal, user_id, beta, st.session_state.ratings_session, cuisine_df)

            vector = meal_nutrition(meal, serving_size, cuisine_df, ing_df)
            meal_protein = vector[0]
            meal_phenyl = vector[1]
            meal_calories = vector[2]
            # st.write(meal_protein, meal_phenyl, meal_calories)
            # st.write(f"Protein: {proteinM} Phenyl: {phe1M} {phe2M} Energy {e1M} {e2M}")

            hs_score = health_safety_score(meal_protein, meal_phenyl, meal_calories, proteinM, phe1M, phe2M, e1M, e2M)
            f_score = final_score(hscore, hs_score, pref_scale)
            st.write(f'Content Score: {cscore:.2f}, Collab Score: {colscore:.2f}, Hybrid Score: {hscore:.2f}, Health Safety Score: {hs_score:.2f}, Final Score: {f_score}')

            labels = ['Phenylalanine', 'Protein', 'Energy']
            user_limits = [100, 100, 100]
            meal_values = []
            vec = meal_nutrition(meal, serving_size, cuisine_df, ing_df)
            meal_protein, meal_phenyl, meal_calories = vec

            def percent(value, limit):
                return round((value / limit) * 100, 1) if limit else 0
            
            phe = st.session_state['phe']
            st.write(st.session_state['phe'])
            st.write(meal_phenyl)

            st.write(phe)
            percent_phe = percent(meal_phenyl, st.session_state['phe'])
            percent_protein = percent(meal_protein, st.session_state['protein'])
            percent_calories = percent(meal_calories, st.session_state['energy'])

            meal_values = [percent_phe, percent_protein, percent_calories]



            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r = user_limits,
                theta = labels,
                fill='toself',
                name='Daily Limit'
            ))
            fig.add_trace(go.Scatterpolar(
                r=meal_values,
                theta=labels,
                fill='toself',
                name='Selected Meal'
            ))

            fig.update_layout(
                polar = dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend = True,
                title = "Meal vs. Daily Nutrient Limits"
            )

            st.plotly_chart(fig)



with tabs[4]:
    st.header('Chat with PKU AI Assistant')
    if chat_df is not None:
        if 'chat_history' not in st.session_state: st.session_state.chat_history=[]
        for msg in st.session_state.chat_history: st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")
        with st.form('chat_form', clear_on_submit=True):
            chat_input = st.text_input('Ask PKU AI:', key='chat_input')
            send = st.form_submit_button('Send')
        if send and chat_input:
            st.session_state.chat_history.append({'role':'user','content':chat_input})
            genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash')
            resp = model.generate_content(chat_input)
            st.session_state.chat_history.append({'role':'assistant','content':resp.text})
    else:
        st.info('Upload Chat Ingredient CSV in the sidebar')

#----------------------------------------------------------------------- Bella TAB 5 :)
with tabs[5]:
    st.header("Rule-Based Meal Recommendation")

    if cuisine_df is None or ing_df is None:
        st.warning("Please upload Cuisine and Ingredient CSVs in the sidebar.")
        st.stop()

    meal_type = st.radio("What type of meal?", ["Breakfast/Snack", "Lunch/Dinner"], key="rule_meal_type")
    portion_ratio = 0.2 if meal_type == "Breakfast/Snack" else 0.3

    # User input to override max daily PHE
    use_manual_phe = st.checkbox("Use custom max daily PHE")
    default_phe = float(st.session_state.get("phe", 0.0))
    if use_manual_phe:
        daily_phe = st.number_input("Enter your max daily PHE (mg):", min_value=0.0, value=default_phe, format="%.2f")
    else:
        daily_phe = default_phe

    meal_phe_target = daily_phe * portion_ratio

    serving_size = st.number_input("Serving size (grams)", min_value=1.0, value=100.0)

    data = []
    for dish in cuisine_df['Meal'].dropna().unique():
        try:
            vec = meal_nutrition(dish, serving_size, cuisine_df, ing_df)
            phe, protein, energy = vec[1], vec[0], vec[2]
            phe_per_gram = phe / serving_size if serving_size else 1
            grams_allowed = meal_phe_target / phe_per_gram if phe_per_gram else 0
            data.append({
                "Dish": dish,
                "Grams Allowed": round(grams_allowed, 1),
                "Phenylalanine (mg)": round(phe, 1),
                "Protein (g)": round(protein, 1),
                "Calories": round(energy, 1)
            })
        except Exception:
            continue

    if data:
        df_recommend = pd.DataFrame(data).sort_values("Grams Allowed", ascending=False)
        st.subheader(f"Recommended Dishes for {meal_type} (max {meal_phe_target:.0f} mg PHE)")
        st.dataframe(df_recommend.reset_index(drop=True))

        fig = px.bar(df_recommend.head(10), x="Dish", y="Grams Allowed",
                     title="Top 10 Dishes You Can Eat by Portion Size",
                     labels={"Grams Allowed": "Grams"})
        st.plotly_chart(fig)
    else:
        st.info("No valid dishes found for calculation.")

    # Single dish calculator section
    st.markdown("---")
    st.subheader("Single Dish Portion Calculator")

    selected_meal_type = st.selectbox("Select Meal Type", ["Breakfast", "Snack", "Lunch", "Dinner"], key="single_meal_type")
    meal_ratio_map = {"Breakfast": 0.2, "Snack": 0.2, "Lunch": 0.3, "Dinner": 0.3}
    single_portion_ratio = meal_ratio_map[selected_meal_type]
    single_meal_phe_target = daily_phe * single_portion_ratio

    all_dishes = sorted(cuisine_df['Meal'].dropna().unique())
    selected_dish = st.selectbox("Select Dish", all_dishes, key="single_dish")

    try:
        vec = meal_nutrition(selected_dish, serving_size, cuisine_df, ing_df)
        protein, phe, energy = vec[0], vec[1], vec[2]
        phe_per_gram = phe / serving_size if serving_size else 1
        grams_allowed = single_meal_phe_target / phe_per_gram if phe_per_gram else 0

        st.success(f"You can eat up to **{grams_allowed:.1f}g** of **{selected_dish}** for {selected_meal_type} (target {single_meal_phe_target:.0f} mg PHE)")
        st.write(f"PHE per {serving_size:.0f}g: {phe:.1f} mg")
        st.write(f"Protein per {serving_size:.0f}g: {protein:.1f} g")
        st.write(f"Calories per {serving_size:.0f}g: {energy:.1f} kcal")

    except KeyError as e:
        st.error(f"Missing data: {e}. Check if all ingredients in the dish are present in the ingredient file.")
    except Exception as e:
        st.error(f"Could not calculate dish nutrition: {e}")

# ----- end of the rule req tab 5

# Gemini
st.divider()
st.subheader("💬 Chat with PKU AI Assistant")

# ✅ Configure Gemini with updated model name
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as config_error:
    st.error("❌ Failed to configure Gemini API.")
    st.error(f"Configuration error: {config_error}")
    st.stop()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI Assistant:** {message['content']}")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Input field ---
# Use form to handle Enter key properly
with st.form("chat_form", clear_on_submit=True):
    chat_input = st.text_input("Ask a question about PKU nutrition:", key="chat_input_form")
    send_clicked = st.form_submit_button("📤 Send Message", use_container_width=True)

# Handle form submission
if send_clicked and chat_input.strip():
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": chat_input})

    with st.spinner("Thinking..."):
        try:
            # Build conversation context
            conversation_context = "You are a helpful assistant specialized in low-phenylalanine diets and PKU nutrition.\n"

            # Add CSV data context if available
            if uploaded_file and 'df' in locals():
                conversation_context += "\nAvailable PKU food data from uploaded CSV:\n"

                # For questions asking for recommendations or comparisons, include more data
                question_lower = chat_input.lower()
                if any(word in question_lower for word in ['recommend', 'lowest', 'highest', 'compare', 'list', 'show', 'all', 'foods']):
                    # Include full food list with nutritional data
                    conversation_context += "Complete food database:\n"
                    for _, row in df.iterrows():
                        conversation_context += f"{row['Food']}: Phe {row['PHE(mg)']}mg, Protein {row['Protein(g)']}g, Energy {row['Energy(kcal)']}kcal\n"
                else:
                    # Just show sample for general questions
                    conversation_context += f"Foods in database: {', '.join(df['Food'].head(10).tolist())}"
                    if len(df) > 10:
                        conversation_context += f" and {len(df)-10} more foods"

                conversation_context += f"\nTotal foods in database: {len(df)}\n"
                conversation_context += "Each food has data for: Phenylalanine (mg), Protein (g), Energy (kcal)\n"

            conversation_context += "\nConversation history:\n"

            # Include recent conversation history (last 10 messages to avoid token limits)
            recent_history = st.session_state.chat_history[-10:]
            for msg in recent_history[:-1]:  # Exclude the current message
                if msg["role"] == "user":
                    conversation_context += f"User: {msg['content']}\n"
                else:
                    conversation_context += f"Assistant: {msg['content']}\n"

            conversation_context += f"\nCurrent question: {chat_input}"

            # Add specific food data if the question mentions foods from the CSV
            if uploaded_file and 'df' in locals():
                # Check if the question mentions any specific foods
                mentioned_foods = []
                chat_lower = chat_input.lower()
                for food in df['Food']:
                    if food.lower() in chat_lower:
                        mentioned_foods.append(food)

                # If specific foods are mentioned, add their detailed data
                if mentioned_foods:
                    conversation_context += f"\n\nDetailed nutritional data for mentioned foods:\n"
                    for food in mentioned_foods:
                        food_data = df[df['Food'] == food].iloc[0]
                        conversation_context += f"{food}: Phenylalanine {food_data['PHE(mg)']}mg, Protein {food_data['Protein(g)']}g, Energy {food_data['Energy(kcal)']}kcal\n"

            response = model.generate_content(conversation_context)

            # Add AI response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response.text})

            # Rerun to show new messages
            st.rerun()

        except Exception as e:
            st.error("❌ Gemini API error.")
            st.error(f"Error details: {str(e)}")
            st.info("💡 Try checking if your API key is valid and has access to Gemini models.")