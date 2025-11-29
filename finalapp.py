import streamlit as st
import pandas as pd
from datetime import datetime, date

# -------------------------------------------------------
# App config
# -------------------------------------------------------
st.set_page_config(page_title="PKU Diet Manager", layout="wide")

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

@st.cache_data
def load_consolidated_foods():
    """
    Load consolidated nutrient database, with flexible column handling.
    Accepts schemas like:
    - name or Ingredient
    - PHE(mg) or phe_mg_per_100g
    - Protein(g) or protein_g_per_100g
    - Energy(kcal) or calories_per_100g
    Optional:
    - Serving_Size(g)
    - piece_weight_g
    """
    try:
        df = pd.read_csv("consolidated_chat_ingredients.csv")
        # Normalize columns
        cols_norm = {c: c.strip() for c in df.columns}
        df = df.rename(columns=cols_norm)

        # Ensure 'name' exists
        if "name" not in df.columns and "Ingredient" in df.columns:
            df = df.rename(columns={"Ingredient": "name"})

        # If per-100g columns exist, convert to common schema with Serving_Size(g)=100
        if "phe_mg_per_100g" in df.columns and "Protein(g)" not in df.columns:
            df["PHE(mg)"] = df["phe_mg_per_100g"]
            df["Protein(g)"] = df.get("protein_g_per_100g", 0.0)
            df["Energy(kcal)"] = df.get("calories_per_100g", 0.0)
            df["Serving_Size(g)"] = 100.0

        # If only per-serving columns exist, make sure Serving_Size(g) is present
        if "Serving_Size(g)" not in df.columns:
            # Fall back to 100g to allow proportional scaling
            df["Serving_Size(g)"] = 100.0

        # Make sure numeric columns are numeric
        for c in ["PHE(mg)", "Protein(g)", "Energy(kcal)", "Serving_Size(g)", "piece_weight_g"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        st.warning(f"Could not load consolidated_chat_ingredients.csv: {e}")
        return pd.DataFrame()

consolidated_db = load_consolidated_foods()

# -------------------------------------------------------
# Static food lists for fallback
# -------------------------------------------------------

BABY_FOODS = {
    "Vegetables": {
        "Carrots, cooked": {"weight_g": 39, "phe_mg": 14, "protein_g": 0.4, "calories": 18},
        "Cauliflower, cooked": {"weight_g": 23, "phe_mg": 16, "protein_g": 0.4, "calories": 6},
    },
    "Fruits": {
        "Applesauce, sweetened": {"weight_g": 86, "phe_mg": 5, "protein_g": 0.2, "calories": 48},
    },
}

TABLE_FOODS = {
    "Vegetables": {
        "Broccoli, cooked (2 Tbsp)": {"weight_g": 20, "phe_mg": 18, "protein_g": 0.6, "calories": 6},
        "Butternut squash, mashed (2 Tbsp)": {"weight_g": 30, "phe_mg": 15, "protein_g": 0.4, "calories": 12},
        "Zucchini (summer squash), cooked (1/4 cup)": {"weight_g": 45, "phe_mg": 15, "protein_g": 0.4, "calories": 9},
    },
    "Fruits": {
        "Banana, sliced (3 Tbsp)": {"weight_g": 42, "phe_mg": 16, "protein_g": 0.4, "calories": 39},
        "Watermelon, cubed (3/4 cup)": {"weight_g": 120, "phe_mg": 18, "protein_g": 0.7, "calories": 38},
    },
    "Breads/Cereals": {
        "Rice (prepared), 2 Tbsp": {"weight_g": 25, "phe_mg": 32, "protein_g": 0.6, "calories": 27},
        "Macaroni (cooked), 1 Tbsp + 1.5 tsp": {"weight_g": 12, "phe_mg": 31, "protein_g": 0.6, "calories": 18},
        "Corn Flakes (1/3 cup)": {"weight_g": 7, "phe_mg": 31, "protein_g": 0.6, "calories": 29},
    }
}

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

def calculate_phenex_amount(protein_needed_g, protein_from_food_g, age_months, product_name="Phenex-2", protein_per_100g=None, calories_per_100g=None):
    """Calculate grams of medical food powder needed; defaults align with Phenex-1/2 if densities not provided."""
    protein_from_medical = max(0.0, protein_needed_g - protein_from_food_g)
    # Defaults if custom is not provided
    if protein_per_100g is None or calories_per_100g is None:
        if age_months < 24:
            product_name = "Phenex-1"
            protein_per_100g = 15
            calories_per_100g = 480
        else:
            if product_name == "Phenex-1":
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
    """Compute milk volume to meet remaining PHE target (per 100mL constants)."""
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

UNIT_TO_GRAMS = {
    "g": 1.0,
    "kg": 1000.0,
    "mg": 0.001,
    "ml": 1.0,     # water-like fallback
    "cup": 240.0,  # generic fallback
    "tbsp": 15.0,
    "tsp": 5.0,
    "piece": None, # requires piece weight
}

def normalize_name(s):
    return str(s).strip().lower()

def parse_portion(text):
    """
    Parse amount strings like:
    '200 g', '1 cup', '2 pieces', '150 ml', '2 tsp'
    Returns (grams_ml, raw_qty, unit). For 'piece' w/o weight, grams_ml=None.
    """
    if not isinstance(text, str):
        return None, None, None
    s = text.strip().lower()
    if not s:
        return None, None, None
    tokens = s.split()
    try:
        qty = float(tokens[0])
        unit = tokens[1] if len(tokens) > 1 else "g"
    except Exception:
        return None, None, None
    if unit not in UNIT_TO_GRAMS:
        unit = "g"
    factor = UNIT_TO_GRAMS[unit]
    if factor is None:
        return None, qty, unit
    return qty * factor, qty, unit

def select_best_match(name, food_db):
    """Exact -> startswith -> contains mapping on 'name' column."""
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

def scale_nutrients(row, weight_g, default_piece_weight_g=None):
    """
    Scale nutrients for a given weight using consolidated CSV schema:
    name, PHE(mg), Protein(g), Energy(kcal), Serving_Size(g)
    """
    if weight_g is None:
        weight_g = default_piece_weight_g or 0.0

    serving_size = row.get("Serving_Size(g)", 100.0)
    phe_serv = row.get("PHE(mg)", 0.0)
    prot_serv = row.get("Protein(g)", 0.0)
    cal_serv = row.get("Energy(kcal)", 0.0)

    # Normalize to per gram
    phe_per_g = (phe_serv / serving_size) if serving_size else 0.0
    prot_per_g = (prot_serv / serving_size) if serving_size else 0.0
    cal_per_g = (cal_serv / serving_size) if serving_size else 0.0

    return {
        "weight_g": weight_g,
        "phe_mg": phe_per_g * weight_g,
        "protein_g": prot_per_g * weight_g,
        "calories": cal_per_g * weight_g,
    }

def load_cuisine_csv(file):
    """
    Accept CSV in various schemas; standardize to:
      - dish (from 'Meal')
      - ingredient (from 'Ingredient')
      - amount (from 'Grams' or 'Weight (g)' converted to 'NN g', or raw 'amount' text)
      - optional 'Meal Type' (default 'Unspecified')
      - optional 'ingredient_piece_weight_g'
    """
    df = pd.read_csv(file)
    # Rename common headers to target names
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "meal": rename_map[c] = "dish"
        elif cl == "ingredient": rename_map[c] = "ingredient"
        elif cl in ["grams", "weight (g)"]: rename_map[c] = "amount"
        elif cl == "serving size (g)": pass
        elif cl == "meal type": pass
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Meal Type" not in df.columns:
        df["Meal Type"] = "Unspecified"

    # If 'amount' is numeric grams, convert to "NN g"
    if "amount" in df.columns and pd.api.types.is_numeric_dtype(df["amount"]):
        df["amount"] = df["amount"].apply(lambda x: f"{x} g")

    required = {"dish", "ingredient", "amount"}
    if not required.issubset(set(df.columns)):
        st.error("Cuisine CSV must include columns: dish, ingredient, amount")
        return pd.DataFrame()
    return df

def compute_cuisine_dishes(cuisine_df, food_db):
    """
    For each dish:
      - Map ingredients to consolidated DB
      - Scale per portion
      - Sum totals (PHE, protein, calories, weight_g)
    Returns dict[dish_name] = {'meal_type', 'ingredients': [...], 'totals': {...}}
    """
    dishes = {}
    if cuisine_df.empty or "name" not in food_db.columns:
        return dishes

    for _, r in cuisine_df.iterrows():
        dish = str(r["dish"]).strip()
        ing_name = str(r["ingredient"]).strip()
        amount_text = str(r["amount"]).strip()
        meal_type = r["Meal Type"] if "Meal Type" in cuisine_df.columns else "Unspecified"

        piece_weight = None
        if "ingredient_piece_weight_g" in cuisine_df.columns:
            try:
                piece_weight = float(r["ingredient_piece_weight_g"])
            except Exception:
                piece_weight = None

        weight_g, raw_qty, unit = parse_portion(amount_text)
        match = select_best_match(ing_name, food_db)

        if match is None:
            scaled = {
                "name": ing_name,
                "unit": unit,
                "raw_qty": raw_qty,
                "weight_g": weight_g if weight_g is not None else (piece_weight or 0.0),
                "phe_mg": 0.0,
                "protein_g": 0.0,
                "calories": 0.0,
                "note": "Ingredient not found in DB",
            }
        else:
            default_piece_weight_g = piece_weight or match.get("piece_weight_g", None) if unit == "piece" and weight_g is None else None
            sv = scale_nutrients(match, weight_g, default_piece_weight_g)
            scaled = {"name": match["name"], "unit": unit, "raw_qty": raw_qty, **sv}

        if dish not in dishes:
            dishes[dish] = {"meal_type": meal_type, "ingredients": [], "totals": {"phe_mg": 0.0, "protein_g": 0.0, "calories": 0.0, "weight_g": 0.0}}

        dishes[dish]["ingredients"].append(scaled)
        tot = dishes[dish]["totals"]
        tot["phe_mg"] += scaled["phe_mg"]
        tot["protein_g"] += scaled["protein_g"]
        tot["calories"] += scaled["calories"]
        tot["weight_g"] += scaled["weight_g"]

    return dishes

# -------------------------------------------------------
# Display functions
# -------------------------------------------------------

def display_child_adult_plan_with_foods(needs, age_months, weight_kg, selected_foods, medical_product_name="Phenex-2", custom_protein_per_100g=None, custom_calories_per_100g=None):
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
        st.subheader("🍽️ Selected foods and dishes")
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
    st.subheader("🥤 Medical food requirement")
    phenex = calculate_phenex_amount(
        needs['protein_g'],
        total_food_protein,
        age_months,
        product_name=medical_product_name,
        protein_per_100g=custom_protein_per_100g,
        calories_per_100g=custom_calories_per_100g
    )
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

    st.info("Mix powder with water; chill to improve taste. You may flavor with allowed fruits or flavoring.")

    # Totals
    st.markdown("---")
    st.subheader("📈 Daily nutrition totals")
    total_protein = total_food_protein + phenex['protein_g']
    total_calories = total_food_calories + phenex['calories_kcal']

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total protein", f"{total_protein:.1f} g", f"Target: {needs['protein_g']:.0f} g")
    with c2:
        phe_ok = needs['phe_mg_min'] <= total_food_phe <= needs['phe_mg_max']
        st.metric("Total PHE", f"{total_food_phe:.0f} mg", f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg" + (" ✅" if phe_ok else " ⚠️"))
    with c3:
        st.metric("Total calories", f"{total_calories:.0f} kcal", f"Target: {needs['energy_kcal']:.0f} kcal")

    remaining_cal = needs['energy_kcal'] - total_calories
    if remaining_cal > 500:
        st.warning(
            f"⚠️ Additional {remaining_cal:.0f} kcal needed.\n"
            "- Add vegetable oils (120 kcal/Tbsp)\n"
            "- Low-protein breads and pastas\n"
            "- PKU-safe fruits and vegetables"
        )

def display_baby_diet_plan_with_solids(result, weight_kg):
    st.markdown("---")
    st.header("📋 Daily Diet Plan")

    needs = result['needs']
    st.subheader(f"Nutritional Targets ({needs['age_group']})")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Protein", f"{needs['protein_g']:.1f} g")
    with c2:
        st.metric("PHE Range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
    with c3:
        st.metric("Calories", f"{needs['energy_kcal']:.0f} kcal")

    # Solid foods
    if result['solid_foods']['foods']:
        st.markdown("### 🍽️ Solid foods (beikost)")
        for food in result['solid_foods']['foods']:
            with st.expander(f"{food['meal']}: {food['name']}"):
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.write(f"**Amount:** {food['weight_g']:.0f} g")
                    st.write(f"**PHE:** {food['phe_mg']:.0f} mg")
                with cc2:
                    st.write(f"**Protein:** {food['protein_g']:.1f} g")
                    st.write(f"**Calories:** {food['calories']:.0f} kcal")

        st.markdown(
            f"**Total from solids:** PHE {result['solid_foods']['total_phe_mg']:.0f} mg | "
            f"Protein {result['solid_foods']['total_protein_g']:.1f} g | "
            f"Calories {result['solid_foods']['total_calories']:.0f} kcal"
        )
        st.markdown("---")

    # Milk + Phenex
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {result['milk']['milk_type']}")
        st.markdown(f"- **{result['milk']['milk_ml']:.0f} mL** per day")
        st.markdown(f"- PHE: {result['milk']['phe_mg']:.0f} mg")
        st.markdown(f"- Protein: {result['milk']['protein_g']:.1f} g")
        st.markdown(f"- Calories: {result['milk']['calories_kcal']:.0f} kcal")
    with c2:
        st.markdown(f"#### {result['phenex']['product_name']}")
        st.markdown(f"- **{result['phenex']['grams']:.1f} g** powder per day")
        st.markdown(f"- Protein: {result['phenex']['protein_g']:.1f} g")
        st.markdown(f"- Calories: {result['phenex']['calories_kcal']:.0f} kcal")
        st.markdown(f"- No PHE")

    st.markdown("---")
    st.subheader("📈 Daily totals")
    totals = result['totals']
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total protein", f"{totals['protein_g']:.1f} g", f"Target: {needs['protein_g']:.1f} g")
    with c2:
        in_range = needs['phe_mg_min'] <= totals['phe_mg'] <= needs['phe_mg_max']
        st.metric("Total PHE", f"{totals['phe_mg']:.0f} mg", f"Range: {needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg" + (" ✅" if in_range else " ⚠️"))
    with c3:
        st.metric("Total calories", f"{totals['calories_kcal']:.0f} kcal", f"Target: {needs['energy_kcal']:.0f} kcal")

# -------------------------------------------------------
# Baby diet computation
# -------------------------------------------------------

def calculate_baby_diet_with_solids(age_months, weight_kg, current_phe_mg_dl, milk_type, solid_foods):
    needs = get_infant_daily_needs(age_months, weight_kg)

    total_solid_phe = sum(food['phe_mg'] for food in solid_foods)
    total_solid_protein = sum(food['protein_g'] for food in solid_foods)
    total_solid_calories = sum(food['calories'] for food in solid_foods)

    phe_target = needs['phe_mg_max']  # conservative target
    remaining_phe = phe_target - total_solid_phe
    if remaining_phe < 0:
        remaining_phe = max(needs['phe_mg_min'] - total_solid_phe, 0)
        st.warning(f"Solid foods PHE ({total_solid_phe:.0f} mg) near/exceeding target; adjusting milk PHE to stay within range.")

    milk = calculate_milk_amount(remaining_phe, milk_type)
    total_phe = total_solid_phe + milk['phe_mg']
    if total_phe > needs['phe_mg_max']:
        safe_remaining = max(needs['phe_mg_max'] - total_solid_phe, 0)
        milk = calculate_milk_amount(safe_remaining, milk_type)

    total_protein_food_milk = total_solid_protein + milk['protein_g']
    phenex = calculate_phenex_amount(needs['protein_g'], total_protein_food_milk, age_months)

    result = {
        'needs': needs,
        'solid_foods': {
            'foods': solid_foods,
            'total_phe_mg': total_solid_phe,
            'total_protein_g': total_solid_protein,
            'total_calories': total_solid_calories
        },
        'milk': milk,
        'phenex': phenex,
        'totals': {
            'protein_g': total_protein_food_milk + phenex['protein_g'],
            'phe_mg': total_solid_phe + milk['phe_mg'],
            'calories_kcal': total_solid_calories + milk['calories_kcal'] + phenex['calories_kcal']
        }
    }
    return result

# -------------------------------------------------------
# Type-ahead ingredient selection for manual add
# -------------------------------------------------------

def typeahead_food_selector(food_db):
    st.markdown("#### Add a food by name")
    query = st.text_input("Type to search (e.g., 'banana', 'macaroni')", "")
    suggestions = []
    if query and not food_db.empty and "name" in food_db.columns:
        q = query.strip().lower()
        # top N suggestions with startswith priority
        starts = food_db[food_db["name"].str.lower().str.startswith(q)]["name"].tolist()
        contains = food_db[food_db["name"].str.lower().str.contains(q)]["name"].tolist()
        # merge unique, prioritize starts
        suggestions = list(dict.fromkeys(starts + contains))[:20]

    if suggestions:
        choice = st.selectbox("Suggestions:", suggestions)
    else:
        choice = None

    portion = st.text_input("Amount (e.g., '100 g', '1 cup', '2 tsp', '1 piece')", "100 g")
    meal = st.selectbox("Meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
    add_btn = st.button("➕ Add selected food")

    if add_btn:
        name_to_use = choice or query
        if not name_to_use:
            st.error("Please type a food name.")
            return None
        match = select_best_match(name_to_use, food_db)
        if match is None:
            st.error("Food not found in DB. Please refine your search.")
            return None
        weight_g, raw_qty, unit = parse_portion(portion)
        default_piece_weight = match.get("piece_weight_g", None) if unit == "piece" and weight_g is None else None
        sv = scale_nutrients(match, weight_g, default_piece_weight)
        return {
            "meal": meal,
            "name": match["name"],
            "weight_g": sv["weight_g"],
            "phe_mg": sv["phe_mg"],
            "protein_g": sv["protein_g"],
            "calories": sv["calories"],
        }
    return None

# -------------------------------------------------------
# Main app
# -------------------------------------------------------

def main():
    # Session state
    if 'profile_created' not in st.session_state:
        st.session_state.profile_created = False
    if 'solid_foods_list' not in st.session_state:
        st.session_state.solid_foods_list = []
    if 'child_adult_foods_list' not in st.session_state:
        st.session_state.child_adult_foods_list = []
    if 'medical_product_name' not in st.session_state:
        st.session_state.medical_product_name = "Phenex-2"
    if 'custom_protein_per_100g' not in st.session_state:
        st.session_state.custom_protein_per_100g = None
    if 'custom_calories_per_100g' not in st.session_state:
        st.session_state.custom_calories_per_100g = None

    if not st.session_state.profile_created:
        st.title("PKU Diet Management System")
        st.markdown("Plan safe PKU diets with foods and medical formula, using portion-controlled nutrient calculations.")

        st.markdown("---")
        st.header("Create profile")

        age_category = st.radio("Profile type:", ["Baby (0-12 months)", "Child (1-12 years)", "Adult (12+ years)"])
        sex = st.radio("Sex:", ["Male", "Female"]) if age_category != "Baby (0-12 months)" else "Male"

        col1, col2 = st.columns(2)
        with col1:
            units = st.radio("Units:", ["Metric", "Imperial"])
            if units == "Metric":
                weight = st.number_input('Weight (kg):', min_value=0.0, step=0.1)
                height_cm = st.number_input('Height (cm):', min_value=0.0, step=1.0)
            else:
                weight_lbs = st.number_input('Weight (lbs):', min_value=0.0, step=0.1)
                height_in = st.number_input('Height (in):', min_value=0.0, step=0.5)
                weight = weight_lbs * 0.453592
                height_cm = height_in * 2.54
        with col2:
            birth_year = st.number_input('Birth year:', min_value=1900, max_value=datetime.now().year, value=2017)
            birth_month = st.number_input('Birth month:', min_value=1, max_value=12, value=1)
            birth_day = st.number_input('Birth day:', min_value=1, max_value=31, value=1)
            current_phe = st.number_input('Current blood PHE (mg/dL):', min_value=0.0, step=0.1)

        milk_type = None
        if age_category == "Baby (0-12 months)":
            milk_type = st.radio("Milk type:", ["Breast Milk (Human Milk)", "Similac With Iron"])

        if st.button("Calculate diet plan", type="primary"):
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
            st.session_state.child_adult_foods_list = []
            st.rerun()

        age_months = calculate_age_months(st.session_state.user_birth_year, st.session_state.user_birth_month, st.session_state.user_birth_day)
        st.title("PKU Diet Plan")

        # Medical food selector
        with st.expander("🥫 Medical food selection", expanded=True):
            product = st.selectbox("Product:", ["Phenex-1", "Phenex-2", "Custom"])
            if product == "Custom":
                cp = st.number_input("Protein per 100 g (g):", min_value=1.0, value=30.0)
                cc = st.number_input("Calories per 100 g (kcal):", min_value=1.0, value=400.0)
                st.session_state.custom_protein_per_100g = cp
                st.session_state.custom_calories_per_100g = cc
            else:
                st.session_state.custom_protein_per_100g = None
                st.session_state.custom_calories_per_100g = None
            st.session_state.medical_product_name = product

        # Baby flow
        if st.session_state.user_age_category == "Baby (0-12 months)":
            if age_months >= 6:
                st.info("Baby is ≥6 months — you can add solid foods.")
                with st.expander("🍽️ Add solid foods", expanded=True):
                    meal_type = st.selectbox("Meal:", ["Breakfast", "Lunch", "Dinner", "Snack"])
                    available_foods = BABY_FOODS if age_months < 9 else {**BABY_FOODS, **TABLE_FOODS}
                    category = st.selectbox("Food category:", list(available_foods.keys()))
                    food_name = st.selectbox("Food:", list(available_foods[category].keys()))
                    fd = available_foods[category][food_name]
                    st.write(f"Standard: {fd['weight_g']} g | {fd['phe_mg']} mg PHE | {fd['protein_g']} g protein | {fd['calories']} kcal")
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
                        st.success(f"Added {servings}× {food_name}")
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

            # Compute and display plan
            baby_result = calculate_baby_diet_with_solids(
                age_months,
                st.session_state.user_weight,
                st.session_state.user_current_phe,
                st.session_state.user_milk_type if st.session_state.user_milk_type else "Breast Milk (Human Milk)",
                st.session_state.solid_foods_list
            )
            display_baby_diet_plan_with_solids(baby_result, st.session_state.user_weight)

        # Child/Adult flow
        else:
            needs = get_child_adult_daily_needs(age_months, st.session_state.user_weight, st.session_state.user_sex)

            # Cuisine CSV upload and mapping
            st.markdown("---")
            st.subheader("🍽️ Upload cuisine recipes (CSV)")
            uploaded = st.file_uploader("Upload a cuisine CSV", type=["csv"])
            if uploaded is not None:
                if consolidated_db.empty:
                    st.error("Consolidated food database not loaded. Please ensure consolidated_chat_ingredients.csv is available.")
                else:
                    cuisine_df = load_cuisine_csv(uploaded)
                    if not cuisine_df.empty:
                        dishes = compute_cuisine_dishes(cuisine_df, consolidated_db)
                        if dishes:
                            st.success(f"Processed {len(dishes)} dish(es).")
                            for dish_name, data in dishes.items():
                                with st.expander(f"Dish: {dish_name}"):
                                    st.write(f"Meal type: {data['meal_type']}")
                                    st.markdown("**Ingredients:**")
                                    for ing in data["ingredients"]:
                                        line = f"- {ing['name']} | {ing['raw_qty']} {ing['unit']} | {ing['weight_g']:.0f} g"
                                        line += f" — PHE {ing['phe_mg']:.0f} mg, Protein {ing['protein_g']:.2f} g, Calories {ing['calories']:.0f} kcal"
                                        if ing.get("note"):
                                            line += f" ({ing['note']})"
                                        st.write(line)
                                    tot = data["totals"]
                                    st.markdown(f"**Dish totals:** PHE {tot['phe_mg']:.0f} mg | Protein {tot['protein_g']:.1f} g | Calories {tot['calories']:.0f} kcal")
                                    if st.button(f"➕ Add '{dish_name}' to plan", key=f"add_dish_{dish_name}"):
                                        st.session_state.child_adult_foods_list.append({
                                            "meal": data["meal_type"] if data["meal_type"] != "Unspecified" else "Cuisine",
                                            "name": dish_name,
                                            "weight_g": tot["weight_g"],
                                            "phe_mg": tot["phe_mg"],
                                            "protein_g": tot["protein_g"],
                                            "calories": tot["calories"],
                                        })
                                        st.success(f"Added '{dish_name}'")
                                        st.rerun()

            # Manual add with type-ahead search (database-backed)
            st.markdown("---")
            st.subheader("🔎 Quick add from consolidated database")
            if consolidated_db.empty:
                st.info("Consolidated database not loaded; using built-in list below.")
            else:
                added = typeahead_food_selector(consolidated_db)
                if added:
                    st.session_state.child_adult_foods_list.append(added)
                    st.success(f"Added {added['name']}")
                    st.rerun()

            # Manual add (fallback built-in)
            with st.expander("🍽️ Add foods manually (built-in list)", expanded=False):
                meal_type = st.selectbox("Meal:", ["Breakfast", "Lunch", "Dinner", "Snack"], key="cad_meal")
                category = st.selectbox("Food category:", list(TABLE_FOODS.keys()), key="cad_cat")
                food_name = st.selectbox("Food:", list(TABLE_FOODS[category].keys()), key="cad_food")
                fd = TABLE_FOODS[category][food_name]
                st.write(f"Std: {fd['weight_g']} g | {fd['phe_mg']} mg PHE | {fd['protein_g']} g protein | {fd['calories']} kcal")
                servings = st.number_input("Servings:", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key="cad_serv")
                if st.button("➕ Add food", key="cad_add"):
                    entry = {
                        "meal": meal_type,
                        "name": food_name,
                        "weight_g": fd['weight_g'] * servings,
                        "phe_mg": fd['phe_mg'] * servings,
                        "protein_g": fd['protein_g'] * servings,
                        "calories": fd['calories'] * servings,
                    }
                    st.session_state.child_adult_foods_list.append(entry)
                    st.success(f"Added {servings}× {food_name}")
                    st.rerun()

                if st.session_state.child_adult_foods_list:
                    st.markdown("---")
                    st.markdown("### Current foods/dishes")
                    for i, food in enumerate(st.session_state.child_adult_foods_list):
                        c1, c2, c3 = st.columns([3, 2, 1])
                        with c1:
                            st.write(f"{food['meal']}: {food['name']}")
                        with c2:
                            st.write(f"{food['weight_g']:.0f} g | {food['phe_mg']:.0f} mg PHE | {food['protein_g']:.1f} g protein")
                        with c3:
                            if st.button("🗑️", key=f"cad_del_{i}"):
                                st.session_state.child_adult_foods_list.pop(i)
                                st.rerun()
                    if st.button("🗑️ Clear all", key="cad_clear_all"):
                        st.session_state.child_adult_foods_list = []
                        st.rerun()

            # Display child/adult plan (single, bug-free function)
            display_child_adult_plan_with_foods(
                needs,
                age_months,
                st.session_state.user_weight,
                st.session_state.child_adult_foods_list,
                medical_product_name=st.session_state.medical_product_name,
                custom_protein_per_100g=st.session_state.custom_protein_per_100g,
                custom_calories_per_100g=st.session_state.custom_calories_per_100g
            )

        # Info
        st.markdown("---")
        st.header("📖 Important information")
        with st.expander("Understanding your numbers"):
            st.markdown(
                "- Children target blood PHE: 2–5 mg/dL; adults: 2–10 mg/dL\n"
                "- Levels should be checked regularly; frequency varies by age/control\n"
                "- PHE rises with higher dietary PHE or catabolism; adequate energy and protein help stabilize"
            )
        with st.expander("Introducing solid foods"):
            st.markdown(
                "- Start when developmentally ready (often 3–4 months; follow clinic guidance)\n"
                "- Weigh foods in grams for accuracy\n"
                "- Keep records of food intake\n"
                "- Adjust milk and medical formula as solids increase\n"
                "- Monitor blood PHE closely with changes"
            )
        with st.expander("⚠️ When to contact your metabolic clinic"):
            st.markdown(
                "- Blood PHE far above target or undetectable\n"
                "- Poor feeding, weight loss, persistent vomiting/diarrhea\n"
                "- Significant behavior changes\n"
                "- Before major diet changes or new meds"
            )
        st.warning(
            "Important: This app supports planning, but always follow your metabolic team's recommendations. "
            "Never make major diet changes without your doctor/dietitian. Blood PHE monitoring is essential."
        )

if __name__ == "__main__":
    main()