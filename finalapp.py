import streamlit as st
import pandas as pd
from datetime import datetime, date
import os

# -------------------------------------------------------
# App config
# -------------------------------------------------------
st.set_page_config(page_title="PKU Diet Manager", layout="wide")

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

@st.cache_data
def load_consolidated_foods():
    """Load consolidated nutrient database."""
    try:
        df = pd.read_csv("consolidated_chat_ingredients.csv")
        cols_norm = {c: c.strip() for c in df.columns}
        df = df.rename(columns=cols_norm)
        
        if "name" not in df.columns and "Ingredient" in df.columns:
            df = df.rename(columns={"Ingredient": "name"})
        
        for c in ["PHE(mg)", "Protein(g)", "Energy(kcal)", "Serving_Size(g)"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        st.warning(f"Could not load consolidated_chat_ingredients.csv: {e}")
        return pd.DataFrame()

@st.cache_data
def load_cuisine_files():
    """Load all cuisine CSV files from current directory."""
    cuisine_data = {}
    cuisine_files = {
        "African": "african_pku_meals.csv",
        "Central European": "central_european_pku_meals.csv",
        "Chinese": "chinese_pku_meals.csv",
        "Eastern European": "eastern_european_pku_meals.csv",
        "Indian": "indian_pku_meals.csv",
        "Italian": "italian_pku_meals.csv",
        "Japanese": "japanese_pku_meals.csv",
        "Mediterranean": "mediterranean_pku_meals.csv",
        "Mexican": "mexican_pku_meals.csv",
        "Scottish": "scottish_pku_meals.csv"
    }
    
    for cuisine_name, filename in cuisine_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Normalize column names
                df.columns = [c.strip() for c in df.columns]
                
                # Standardize column names
                rename_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if cl == "meal": rename_map[c] = "dish"
                    elif cl == "ingredient": rename_map[c] = "ingredient"
                    elif cl in ["grams", "weight (g)"]: rename_map[c] = "amount"
                
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                # Add Meal Type if missing - for Japanese, Mexican, Mediterranean
                if "Meal Type" not in df.columns:
                    df["Meal Type"] = "ALL"  # Can be used for both BS and LD
                
                cuisine_data[cuisine_name] = df
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    
    return cuisine_data

consolidated_db = load_consolidated_foods()
cuisine_db = load_cuisine_files()

# -------------------------------------------------------
# Helpers: age, targets, phenex
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

def calculate_phenex_amount(protein_needed_g, protein_from_food_g, age_months):
    """Calculate grams of medical food powder needed."""
    protein_from_medical = max(0.0, protein_needed_g - protein_from_food_g)
    
    if age_months < 24:
        product_name = "Phenex-1"
        protein_per_100g = 15
        calories_per_100g = 480
    else:
        product_name = "Phenex-2"
        protein_per_100g = 30
        calories_per_100g = 410
    
    grams = (protein_from_medical * 100.0) / protein_per_100g if protein_per_100g > 0 else 0.0
    return {
        "product_name": product_name,
        "grams": grams,
        "protein_g": (grams/100.0)*protein_per_100g,
        "calories_kcal": (grams/100.0)*calories_per_100g
    }

def calculate_milk_amount(phe_target_mg, milk_type):
    """Compute milk volume to meet remaining PHE target."""
    if milk_type == "Breast Milk (Human Milk)":
        phe_per_100ml = 48
        protein_per_100ml = 1.07
        energy_per_100ml = 72
    else:
        phe_per_100ml = 59
        protein_per_100ml = 1.40
        energy_per_100ml = 68
    ml = (phe_target_mg / phe_per_100ml) * 100.0 if phe_per_100ml > 0 else 0.0
    return {
        "milk_type": milk_type,
        "milk_ml": ml,
        "phe_mg": phe_target_mg,
        "protein_g": (ml/100.0)*protein_per_100ml,
        "calories_kcal": (ml/100.0)*energy_per_100ml
    }

# -------------------------------------------------------
# Ingredient mapping & parsing
# -------------------------------------------------------

def normalize_name(s):
    return str(s).strip().lower()

def parse_portion(text):
    """Parse amount strings like '200 g', '1 cup', etc."""
    if not isinstance(text, str):
        return None
    s = text.strip().lower()
    if not s:
        return None
    tokens = s.split()
    try:
        qty = float(tokens[0])
        return qty  # Return grams
    except Exception:
        return None

def select_best_match(name, food_db):
    """Find best matching ingredient in database."""
    if food_db.empty or "name" not in food_db.columns:
        return None
    target = normalize_name(name)
    exact = food_db[food_db["name"].str.lower() == target]
    if len(exact) == 1:
        return exact.iloc[0]
    sw = food_db[food_db["name"].str.lower().str.startswith(target)]
    if len(sw) >= 1:
        return sw.iloc[0]
    ct = food_db[food_db["name"].str.lower().str.contains(target)]
    if len(ct) >= 1:
        return ct.iloc[0]
    return None

def scale_nutrients(row, weight_g):
    """Scale nutrients for a given weight."""
    serving_size = row.get("Serving_Size(g)", 100.0)
    phe_serv = row.get("PHE(mg)", 0.0)
    prot_serv = row.get("Protein(g)", 0.0)
    cal_serv = row.get("Energy(kcal)", 0.0)
    
    phe_per_g = (phe_serv / serving_size) if serving_size else 0.0
    prot_per_g = (prot_serv / serving_size) if serving_size else 0.0
    cal_per_g = (cal_serv / serving_size) if serving_size else 0.0
    
    return {
        "weight_g": weight_g,
        "phe_mg": phe_per_g * weight_g,
        "protein_g": prot_per_g * weight_g,
        "calories": cal_per_g * weight_g,
    }

def compute_dish_nutrients(dish_df, food_db):
    """Calculate total nutrients for a dish."""
    total_phe = 0.0
    total_protein = 0.0
    total_calories = 0.0
    total_weight = 0.0
    ingredients_list = []
    
    for _, row in dish_df.iterrows():
        ing_name = str(row["ingredient"]).strip()
        amount = parse_portion(str(row["amount"]))
        
        if amount is None:
            continue
        
        match = select_best_match(ing_name, food_db)
        
        if match is None:
            ingredients_list.append({
                "name": ing_name,
                "weight_g": amount,
                "phe_mg": 0.0,
                "protein_g": 0.0,
                "calories": 0.0,
                "note": "Not found in database"
            })
            continue
        
        scaled = scale_nutrients(match, amount)
        ingredients_list.append({
            "name": match["name"],
            **scaled
        })
        
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

# -------------------------------------------------------
# Display functions
# -------------------------------------------------------

def display_diet_plan(needs, selected_foods, age_months):
    st.markdown("---")
    st.header("📋 Daily Diet Plan")
    
    st.subheader(f"Nutritional Targets ({needs['age_group']})")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Protein", f"{needs['protein_g']:.0f} g")
    with c2:
        st.metric("PHE Range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
    with c3:
        st.metric("Calories", f"{needs['energy_kcal']:.0f} kcal")
    
    total_food_phe = sum(f['phe_mg'] for f in selected_foods)
    total_food_protein = sum(f['protein_g'] for f in selected_foods)
    total_food_calories = sum(f['calories'] for f in selected_foods)
    
    if selected_foods:
        st.markdown("---")
        st.subheader("🍽️ Selected meals")
        for food in selected_foods:
            with st.expander(f"{food['meal']}: {food['name']}"):
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.write(f"**Amount:** {food['weight_g']:.0f} g")
                    st.write(f"**PHE:** {food['phe_mg']:.0f} mg")
                with cc2:
                    st.write(f"**Protein:** {food['protein_g']:.1f} g")
                    st.write(f"**Calories:** {food['calories']:.0f} kcal")
        st.markdown(f"**Total from foods:** PHE {total_food_phe:.0f} mg | Protein {total_food_protein:.1f} g | Calories {total_food_calories:.0f} kcal")
    
    st.markdown("---")
    st.subheader("🥤 Recommended medical food")
    phenex = calculate_phenex_amount(needs['protein_g'], total_food_protein, age_months)
    
    cc1, cc2 = st.columns(2)
    with cc1:
        servings = 5
        st.markdown(f"#### {phenex['product_name']}")
        st.markdown(f"- **{phenex['grams']:.1f} g** powder per day")
        st.markdown(f"- Divide into 4–6 servings")
        st.markdown(f"- Per serving: ~{phenex['grams']/servings:.1f} g")
    with cc2:
        st.markdown("#### Provides:")
        st.markdown(f"- Protein: {phenex['protein_g']:.1f} g")
        st.markdown(f"- Calories: {phenex['calories_kcal']:.0f} kcal")
        st.markdown(f"- No PHE")
    
    st.info("💡 Mix powder with water; chill to improve taste. You may flavor with allowed fruits.")
    
    st.markdown("---")
    st.subheader("📈 Daily nutrition totals")
    total_protein = total_food_protein + phenex['protein_g']
    total_calories = total_food_calories + phenex['calories_kcal']
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total protein", f"{total_protein:.1f} g", f"Target: {needs['protein_g']:.0f} g")
    with c2:
        phe_ok = needs['phe_mg_min'] <= total_food_phe <= needs['phe_mg_max']
        st.metric("Total PHE", f"{total_food_phe:.0f} mg", 
                 f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg" + (" ✅" if phe_ok else " ⚠️"))
    with c3:
        st.metric("Total calories", f"{total_calories:.0f} kcal", f"Target: {needs['energy_kcal']:.0f} kcal")
    
    remaining_cal = needs['energy_kcal'] - total_calories
    if remaining_cal > 500:
        st.warning(
            f"⚠️ Additional {remaining_cal:.0f} kcal needed.\n\n"
            "- Add vegetable oils (120 kcal/Tbsp)\n"
            "- Low-protein breads and pastas\n"
            "- PKU-safe fruits and vegetables"
        )

# -------------------------------------------------------
# Main app
# -------------------------------------------------------

def main():
    if 'profile_created' not in st.session_state:
        st.session_state.profile_created = False
    if 'selected_foods_list' not in st.session_state:
        st.session_state.selected_foods_list = []
    
    if not st.session_state.profile_created:
        st.title("PKU Diet Management System")
        st.markdown("Plan safe PKU diets with culturally diverse foods and medical formula.")
        
        st.markdown("---")
        st.header("Create profile")
        
        age_category = st.radio("Profile type:", ["Child (1-12 years)", "Adult (12+ years)"])
        sex = st.radio("Sex:", ["Male", "Female"])
        
        col1, col2 = st.columns(2)
        with col1:
            units = st.radio("Units:", ["Metric", "Imperial"])
            if units == "Metric":
                weight = st.number_input('Weight (kg):', min_value=0.0, step=0.1, value=20.0)
                height_cm = st.number_input('Height (cm):', min_value=0.0, step=1.0, value=120.0)
            else:
                weight_lbs = st.number_input('Weight (lbs):', min_value=0.0, step=0.1, value=44.0)
                height_in = st.number_input('Height (in):', min_value=0.0, step=0.5, value=47.0)
                weight = weight_lbs * 0.453592
                height_cm = height_in * 2.54
        with col2:
            birth_year = st.number_input('Birth year:', min_value=1900, max_value=datetime.now().year, value=2017)
            birth_month = st.number_input('Birth month:', min_value=1, max_value=12, value=1)
            birth_day = st.number_input('Birth day:', min_value=1, max_value=31, value=1)
            current_phe = st.number_input('Current blood PHE (mg/dL):', min_value=0.0, step=0.1, value=5.0)
        
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
            st.session_state.selected_foods_list = []
            st.rerun()
        
        age_months = calculate_age_months(st.session_state.user_birth_year, st.session_state.user_birth_month, st.session_state.user_birth_day)
        needs = get_child_adult_daily_needs(age_months, st.session_state.user_weight, st.session_state.user_sex)
        
        st.title("PKU Meal Planning")
        
        # Cuisine-based meal selection
        st.markdown("---")
        st.header("🌍 Select meals from cuisines")
        
        if not cuisine_db:
            st.error("No cuisine databases loaded. Please ensure cuisine CSV files are in the current directory.")
        else:
            cuisine_choice = st.selectbox("Choose cuisine:", list(cuisine_db.keys()))
            
            if cuisine_choice:
                cuisine_df = cuisine_db[cuisine_choice]
                
                # Determine if this cuisine has meal type differentiation
                has_meal_types = "Meal Type" in cuisine_df.columns and cuisine_df["Meal Type"].nunique() > 1
                no_meal_types = cuisine_choice in ["Japanese", "Mediterranean", "Mexican"]
                
                if has_meal_types and not no_meal_types:
                    meal_category = st.radio("Meal category:", ["Breakfast/Snack", "Lunch/Dinner"])
                    
                    if meal_category == "Breakfast/Snack":
                        filtered_df = cuisine_df[cuisine_df["Meal Type"] == "BS"]
                    else:
                        filtered_df = cuisine_df[cuisine_df["Meal Type"] == "LD"]
                else:
                    st.info(f"All {cuisine_choice} dishes available for any meal")
                    filtered_df = cuisine_df
                
                # Get unique dishes
                unique_dishes = filtered_df["dish"].dropna().unique().tolist()
                
                if unique_dishes:
                    st.markdown("#### Search and select a dish")
                    search_query = st.text_input("Type to search dishes:", "")
                    
                    # Filter dishes based on search
                    if search_query:
                        matching_dishes = [d for d in unique_dishes if search_query.lower() in d.lower()]
                    else:
                        matching_dishes = unique_dishes
                    
                    if matching_dishes:
                        selected_dish = st.selectbox("Available dishes:", matching_dishes)
                        
                        if selected_dish:
                            # Get all rows for this dish
                            dish_df = filtered_df[filtered_df["dish"] == selected_dish]
                            
                            # Calculate nutrients
                            dish_nutrients = compute_dish_nutrients(dish_df, consolidated_db)
                            
                            # Display dish details
                            with st.expander(f"📖 View ingredients for '{selected_dish}'", expanded=True):
                                st.markdown("**Ingredients:**")
                                for ing in dish_nutrients["ingredients"]:
                                    line = f"- {ing['name']}: {ing['weight_g']:.0f} g"
                                    line += f" (PHE: {ing['phe_mg']:.0f} mg, Protein: {ing['protein_g']:.1f} g, Calories: {ing['calories']:.0f} kcal)"
                                    if ing.get("note"):
                                        line += f" ⚠️ {ing['note']}"
                                    st.write(line)
                                
                                st.markdown("---")
                                tot = dish_nutrients["totals"]
                                st.markdown(f"**Dish totals:** PHE {tot['phe_mg']:.0f} mg | Protein {tot['protein_g']:.1f} g | Calories {tot['calories']:.0f} kcal | Weight {tot['weight_g']:.0f} g")
                            
                            meal_type_label = st.selectbox("Assign to meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                            
                            if st.button(f"➕ Add '{selected_dish}' to my plan", type="primary"):
                                st.session_state.selected_foods_list.append({
                                    "meal": meal_type_label,
                                    "name": selected_dish,
                                    "weight_g": dish_nutrients["totals"]["weight_g"],
                                    "phe_mg": dish_nutrients["totals"]["phe_mg"],
                                    "protein_g": dish_nutrients["totals"]["protein_g"],
                                    "calories": dish_nutrients["totals"]["calories"],
                                })
                                st.success(f"✅ Added '{selected_dish}' to your plan!")
                                st.rerun()
                    else:
                        st.info("No dishes match your search. Try different keywords.")
                else:
                    st.warning("No dishes available for this selection.")
        
        # Current meal plan
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
                        st.rerun()
            
            if st.button("🗑️ Clear all meals"):
                st.session_state.selected_foods_list = []
                st.rerun()
        
        # Display diet plan
        display_diet_plan(needs, st.session_state.selected_foods_list, age_months)
        
        # Info section
        st.markdown("---")
        st.header("📖 Important information")
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
        st.warning(
            "⚠️ **Important:** This app supports planning only. Always follow your metabolic team's recommendations. "
            "Never make major diet changes without consulting your doctor/dietitian."
        )

if __name__ == "__main__":
    main()
    