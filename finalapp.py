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
        df.columns = [c.strip().lower() for c in df.columns]
        if "name" not in df.columns and "Ingredient" in df.columns:
            df = df.rename(columns={"Ingredient": "name"})
        # Normalize numerics
        for c in ["PHE(mg)", "Protein(g)", "Energy(kcal)", "Serving_Size(g)"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # Ensure 'name' is string
        if "name" in df.columns:
            df["name"] = df["name"].astype(str)
        return df
    except Exception as e:
        st.warning(f"Could not load consolidated_chat_ingredients.csv: {e}")
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

                # Canonicalize columns
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

                # If Meal Type missing, mark ALL
                if "Meal Type" not in df.columns:
                    df["Meal Type"] = "ALL"
                # Ensure required minimal columns exist
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
# Static baby foods (for fast beikost planning)
# Values reflect typical PKU planning lists (mg PHE, g protein, kcal, weights)
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
# Helpers: age, targets, medical-food gap
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

# Infant ranges adapted from PKU protocol (Table 1-1): PHE mg/kg, protein g/kg, energy kcal/kg, fluid mL/kg
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

# Children/adults daily needs aligned to protocol ranges (Table 1-1); target is midpoint of min/max PHE
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

# Milk nutrient density (per 100 mL) aligned with PKU protocol Table 1-2; used to compute PHE from milk
def calculate_milk_amount(phe_target_mg, milk_type, split_ratio=0.5):
    # Handle None or invalid milk type
    if milk_type is None:
        milk_type = "Breast Milk (Human Milk)"  # Default fallback
    
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
    elif "Both" in str(milk_type):  # Handle "Both" with or without percentages
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
        # Fallback for any other case
        return {"milk_type": "Breast Milk (Human Milk)", "milk_ml": 0,
                "phe_mg": 0, "protein_g": 0, "calories_kcal": 0}

# Compute medical food "gap" (protein and energy only), no product selection shown.
def compute_medical_food_gap(protein_needed_g, protein_from_food_g, age_months):
    """Return protein gap and estimated energy contribution if filled by amino-acid medical food."""
    protein_gap = max(0.0, protein_needed_g - protein_from_food_g)
    # Approximate energy density based on age group mixtures (kcal per g powder)
    # Phenylalanine-free amino acid medical foods typically ~4.1–4.8 kcal/g; we stay neutral here.
    if age_months < 24:
        kcal_per_g_powder = 4.8  # infant blends often ~480 kcal/100 g
        protein_per_100g = 15.0  # g protein equivalent per 100 g
    else:
        kcal_per_g_powder = 4.1  # children/adults blends ~410 kcal/100 g
        protein_per_100g = 30.0  # g protein equivalent per 100 g
    grams_powder = (protein_gap * 100.0) / protein_per_100g if protein_per_100g > 0 else 0.0
    calories_kcal = grams_powder * kcal_per_g_powder
    return {
        "protein_gap_g": protein_gap,
        "estimated_powder_g": grams_powder,
        "estimated_calories_kcal": calories_kcal
    }

# -------------------------------------------------------
# Ingredient mapping & dish nutrient computation
# -------------------------------------------------------

def normalize_name(s):
    return str(s).strip().lower()

def parse_portion(text):
    """Parse simple amount strings that start with a number (interpreted as grams)."""
    if not isinstance(text, str):
        return None
    s = text.strip().lower()
    if not s:
        return None
    tokens = s.split()
    try:
        qty = float(tokens[0])
        return qty
    except Exception:
        return None

def select_best_match(name, food_db):
    """Find best matching ingredient in database."""
    if food_db.empty or "name" not in food_db.columns:
        return None
    target = normalize_name(name)
    names_col = food_db["name"].str.lower()
    exact = food_db[names_col == target]
    if len(exact) == 1:
        return exact.iloc[0]
    sw = food_db[names_col.str.startswith(target)]
    if len(sw) >= 1:
        return sw.iloc[0]
    ct = food_db[names_col.str.contains(target)]
    if len(ct) >= 1:
        return ct.iloc[0]
    return None

def scale_nutrients(row, weight_g):
    """Scale nutrients per ingredient by grams."""
    serving_size = row.get("serving_size(g)", 100.0)
    phe_serv     = row.get("phe(mg)", 0.0)
    prot_serv    = row.get("protein(g)", 0.0)
    cal_serv     = row.get("energy(kcal)", 0.0)

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
    """Calculate total nutrients for a dish from its rows (ingredient, amount)."""
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
            total_weight += amount
            continue
        scaled = scale_nutrients(match, amount)
        ingredients_list.append({"name": match["name"], **scaled})
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
# Baby diet computation (beikost + milk + medical food gap)
# -------------------------------------------------------

def calculate_baby_diet_with_solids(age_months, weight_kg, milk_type, solid_foods):
    needs = get_infant_daily_needs(age_months, weight_kg)
    total_solid_phe = sum(food['phe_mg'] for food in solid_foods)
    total_solid_protein = sum(food['protein_g'] for food in solid_foods)
    total_solid_calories = sum(food['calories'] for food in solid_foods)
    # Aim at upper bound safely, then adjust if over
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
            'protein_g': total_protein_food_milk + medical_gap['protein_gap_g'],  # after gap is filled
            'phe_mg': total_solid_phe + milk['phe_mg'],  # medical food has no PHE
            'calories_kcal': total_solid_calories + milk['calories_kcal'] + medical_gap['estimated_calories_kcal']
        }
    }
    return result

# -------------------------------------------------------
# UI: display functions
# -------------------------------------------------------

def display_baby_diet_plan(result):
    st.markdown("---")
    st.header("📋 Daily diet plan")
    needs = result['needs']
    st.subheader(f"Nutritional targets ({needs['age_group']})")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Protein", f"{needs['protein_g']:.1f} g")
    with c2:
        st.metric("PHE range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
    with c3:
        st.metric("Calories", f"{needs['energy_kcal']:.0f} kcal")

    if result['solid_foods']['foods']:
        st.markdown("---")
        st.markdown("### 🍽️ Solid foods (beikost)")
        for food in result['solid_foods']['foods']:
            with st.expander(f"{food['meal']}: {food['name']}"):
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.write(f"**Amount:** {food['weight_g']:.0f} g")
                    st.write(f"**PHE:** {food['phe_mg']:.0f} mg")
                with cc2:
                    st.write(f"**Protein:** {food['protein_g']:.2f} g")
                    st.write(f"**Calories:** {food['calories']:.0f} kcal")
        st.markdown(
            f"**Total from solids:** PHE {result['solid_foods']['total_phe_mg']:.0f} mg | "
            f"Protein {result['solid_foods']['total_protein_g']:.2f} g | "
            f"Calories {result['solid_foods']['total_calories']:.0f} kcal"
        )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {result['milk']['milk_type']}")
        st.markdown(f"- **{result['milk']['milk_ml']:.0f} mL** per day")
        st.markdown(f"- PHE: {result['milk']['phe_mg']:.0f} mg")
        st.markdown(f"- Protein: {result['milk']['protein_g']:.2f} g")
        st.markdown(f"- Calories: {result['milk']['calories_kcal']:.0f} kcal")
    with c2:
        st.markdown("#### Medical food (needs only)")
        gap = result['medical_food_gap']
        st.markdown(f"- Protein gap to fill: **{gap['protein_gap_g']:.2f} g**")
        st.markdown(f"- Estimated powder: {gap['estimated_powder_g']:.1f} g")
        st.markdown(f"- Estimated calories: {gap['estimated_calories_kcal']:.0f} kcal")
        st.markdown("- Phenylalanine: 0 mg (no PHE)")

    st.markdown("---")
    st.subheader("📈 Daily totals")
    totals = result['totals']
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total protein", f"{totals['protein_g']:.2f} g", f"Target: {needs['protein_g']:.2f} g")
    with c2:
        in_range = needs['phe_mg_min'] <= totals['phe_mg'] <= needs['phe_mg_max']
        st.metric("Total PHE", f"{totals['phe_mg']:.0f} mg",
                  f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg" + (" ✅" if in_range else " ⚠️"))
    with c3:
        st.metric("Total calories", f"{totals['calories_kcal']:.0f} kcal", f"Target: {needs['energy_kcal']:.0f} kcal")

# -------------------------------------------------------
# Custom dish entry
# -------------------------------------------------------

def add_custom_dish_ui(consolidated_db):
    st.markdown("#### Add a custom dish")
    with st.form("custom_dish_form", clear_on_submit=False):
        dish_name = st.text_input("Dish name")
        meal_type_label = st.selectbox("Assign to meal", ["Breakfast", "Lunch", "Dinner", "Snack"])
        st.markdown("Enter ingredients (name and grams). If not found in DB, you may add PHE/protein/kcal per 100 g.")
        ing_cols = st.columns([3, 1, 1, 1, 1])
        # Collect up to 8 ingredients for simplicity
        ingredients = []
        for i in range(8):
            with ing_cols[0]:
                n = st.text_input(f"Ingredient {i+1} name", key=f"c_ing_{i}")
            with ing_cols[1]:
                g = st.number_input(f"g {i+1}", min_value=0.0, step=1.0, value=0.0, key=f"c_g_{i}")
            with ing_cols[2]:
                phe100 = st.number_input(f"PHE/100g {i+1}", min_value=0.0, step=1.0, value=0.0, key=f"c_phe100_{i}")
            with ing_cols[3]:
                pro100 = st.number_input(f"Prot/100g {i+1}", min_value=0.0, step=0.1, value=0.0, key=f"c_pro100_{i}")
            with ing_cols[4]:
                kcal100 = st.number_input(f"kcal/100g {i+1}", min_value=0.0, step=1.0, value=0.0, key=f"c_kcal100_{i}")
            if n and g > 0:
                ingredients.append({"ingredient": n, "amount": g, "phe100": phe100, "pro100": pro100, "kcal100": kcal100})
        submitted = st.form_submit_button("➕ Add custom dish")
    if submitted and dish_name and ingredients:
        # Build a DataFrame mimicking cuisine rows
        rows = []
        for ing in ingredients:
            row = {"dish": dish_name, "ingredient": ing["ingredient"], "amount": ing["amount"], "Meal Type": "ALL"}
            rows.append(row)
        dish_df = pd.DataFrame(rows)
        # Compute nutrients; override for missing ingredients if user supplied per-100g
        totals = {"phe_mg": 0.0, "protein_g": 0.0, "calories": 0.0, "weight_g": 0.0}
        ing_list = []
        for ing in ingredients:
            match = select_best_match(ing["ingredient"], consolidated_db)
            if match is None and (ing["phe100"] > 0 or ing["pro100"] > 0 or ing["kcal100"] > 0):
                # Scale using provided per-100g
                weight_g = ing["amount"]
                phe_mg = (ing["phe100"]/100.0)*weight_g
                protein_g = (ing["pro100"]/100.0)*weight_g
                calories = (ing["kcal100"]/100.0)*weight_g
                record = {"name": ing["ingredient"], "weight_g": weight_g, "phe_mg": phe_mg,
                          "protein_g": protein_g, "calories": calories, "note": "User-supplied values"}
            else:
                if match is None:
                    # Not found and no overrides
                    record = {"name": ing["ingredient"], "weight_g": ing["amount"], "phe_mg": 0.0,
                              "protein_g": 0.0, "calories": 0.0, "note": "Not found"}
                else:
                    record = scale_nutrients(match, ing["amount"])
                    record["name"] = match["name"]
            ing_list.append(record)
            totals["phe_mg"] += record["phe_mg"]
            totals["protein_g"] += record["protein_g"]
            totals["calories"] += record["calories"]
            totals["weight_g"] += record["weight_g"]
        st.success(f"✅ Added custom dish '{dish_name}'")
        return {
            "meal": meal_type_label,
            "name": dish_name,
            "weight_g": totals["weight_g"],
            "phe_mg": totals["phe_mg"],
            "protein_g": totals["protein_g"],
            "calories": totals["calories"],
            "ingredients": ing_list
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

    if not st.session_state.profile_created:
        st.title("PKU Diet Management System")
        st.markdown("Plan safe PKU diets with culturally diverse foods and medical food needs only.")
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
                    # Type-ahead: filter options by query
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

            # Calculate and display baby plan
            baby_result = calculate_baby_diet_with_solids(
                age_months,
                st.session_state.user_weight,
                st.session_state.user_milk_type if st.session_state.user_milk_type else "Breast Milk (Human Milk)",
                st.session_state.solid_foods_list
            )
            display_baby_diet_plan(baby_result)

        # Child/Adult flow (1+ years)
        else:
            needs = get_child_adult_daily_needs(age_months, st.session_state.user_weight, st.session_state.user_sex)

            st.markdown("---")
            st.header("🌍 Select meals from cuisines")

            if not cuisine_db:
                st.error("No cuisine databases loaded. Please ensure cuisine CSV files are in the current directory.")
            else:
                cuisine_choice = st.selectbox("Choose cuisine:", list(cuisine_db.keys()))
                if cuisine_choice:
                    cuisine_df = cuisine_db[cuisine_choice]
                    # BS/LD handling
                    cuisine_without_types = cuisine_choice in ["Japanese", "Mediterranean", "Mexican"]
                    meal_category = st.radio("Meal category:", ["Breakfast/Snack", "Lunch/Dinner"])
                    if not cuisine_without_types and "Meal Type" in cuisine_df.columns and cuisine_df["Meal Type"].nunique() > 1:
                        filtered_df = cuisine_df[cuisine_df["Meal Type"] == ("BS" if meal_category == "Breakfast/Snack" else "LD")]
                    else:
                        # For cuisines without BS/LD, show ALL for both
                        filtered_df = cuisine_df

                    # Type-ahead search for dishes
                    st.markdown("#### Search and select a dish")
                    search_query = st.text_input("Type to search dishes:", "")
                    unique_dishes = filtered_df["dish"].dropna().astype(str).unique().tolist()
                    matching_dishes = [d for d in unique_dishes if search_query.lower() in d.lower()] if search_query else unique_dishes
                    selected_dish = st.selectbox("Available dishes:", matching_dishes) if matching_dishes else None

                    if selected_dish:
                        dish_rows = filtered_df[filtered_df["dish"] == selected_dish]
                        dish_nutrients = compute_dish_nutrients(dish_rows, consolidated_db)

                        with st.expander(f"📖 View ingredients for '{selected_dish}'", expanded=True):
                            st.markdown("**Ingredients:**")
                            for ing in dish_nutrients["ingredients"]:
                                line = f"- {ing['name']}: {ing['weight_g']:.0f} g"
                                line += f" (PHE: {ing['phe_mg']:.0f} mg, Protein: {ing['protein_g']:.2f} g, Calories: {ing['calories']:.0f} kcal)"
                                if ing.get("note"):
                                    line += f" ⚠️ {ing['note']}"
                                st.write(line)
                            st.markdown("---")
                            tot = dish_nutrients["totals"]
                            st.markdown(f"**Dish totals:** PHE {tot['phe_mg']:.0f} mg | Protein {tot['protein_g']:.2f} g | Calories {tot['calories']:.0f} kcal | Weight {tot['weight_g']:.0f} g")

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
                            st.success(f"✅ Added '{selected_dish}' to your plan!")
                            st.rerun()
                    elif search_query:
                        st.info("No dishes match your search. Try different keywords.")

                # Add custom dish
                st.markdown("---")
                if st.button("➕ Add a custom dish"):
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
                        st.rerun()


            # Current meal plan list
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

            # Totals and medical food needs
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

            # Medical food gap only (no picker)
            gap = compute_medical_food_gap(needs['protein_g'], total_food_protein, age_months)
            phe_ok = needs['phe_mg_min'] <= total_food_phe <= needs['phe_mg_max']
            total_protein = total_food_protein + gap['protein_gap_g']
            total_calories = total_food_calories + gap['estimated_calories_kcal']

            st.markdown("#### Medical food (needs only)")
            st.markdown(f"- Protein gap to fill: **{gap['protein_gap_g']:.1f} g**")
            st.markdown(f"- Estimated powder: {gap['estimated_powder_g']:.1f} g")
            st.markdown(f"- Estimated calories: {gap['estimated_calories_kcal']:.0f} kcal")
            st.markdown("- Phenylalanine: 0 mg (no PHE)")

            st.markdown("---")
            st.subheader("📈 Daily nutrition totals")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total protein", f"{total_protein:.1f} g", f"Target: {needs['protein_g']:.0f} g")
            with c2:
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

        # Info
        st.markdown("---")
        st.header("📖 Important information")
        with st.expander("Understanding your numbers"):
            st.markdown(
                "- Children target blood PHE: 2–5 mg/dL; adults: 2–10 mg/dL\n"
                "- Levels should be checked regularly\n"
                "- Adequate energy and protein help stabilize PHE levels"
            )
        with st.expander("Introducing solid foods (babies)"):
            st.markdown(
                "- Start when developmentally ready (often ~6 months; follow clinic guidance)\n"
                "- Weigh foods in grams for accuracy\n"
                "- Keep records of food intake\n"
                "- Adjust milk and medical food as solids increase\n"
                "- Monitor blood PHE closely with changes"
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