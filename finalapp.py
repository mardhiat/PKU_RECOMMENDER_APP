import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import math
import plotly.graph_objects as go

# BABY FOOD DATABASE (from Table 1-3, Gerber Baby Foods)
BABY_FOODS = {
    "Fruits": {
        "Bananas, 2nd Foods": {"weight_g": 47, "phe_mg": 15, "protein_g": 0.5, "calories": 42},
        "Peaches, 2nd Foods": {"weight_g": 88, "phe_mg": 15, "protein_g": 0.6, "calories": 56},
        "Pears, 2nd Foods": {"weight_g": 88, "phe_mg": 15, "protein_g": 0.4, "calories": 65},
        "Applesauce, 2nd Foods": {"weight_g": 86, "phe_mg": 5, "protein_g": 0.2, "calories": 48},
    },
    "Vegetables": {
        "Carrots, 2nd Foods": {"weight_g": 75, "phe_mg": 15, "protein_g": 0.6, "calories": 22},
        "Green beans, 2nd Foods": {"weight_g": 31, "phe_mg": 15, "protein_g": 0.4, "calories": 10},
        "Squash, 2nd Foods": {"weight_g": 33, "phe_mg": 15, "protein_g": 0.2, "calories": 10},
        "Sweet potatoes, 2nd Foods": {"weight_g": 48, "phe_mg": 30, "protein_g": 0.5, "calories": 30},
        "Peas, 2nd Foods": {"weight_g": 24, "phe_mg": 30, "protein_g": 0.7, "calories": 12},
    }
}

# TABLE FOODS for older babies (from Table 1-3, Table Foods)
TABLE_FOODS = {
    "Vegetables": {
        "Broccoli, cooked": {"weight_g": 20, "phe_mg": 18, "protein_g": 0.6, "calories": 6},
        "Squash (butternut), mashed": {"weight_g": 30, "phe_mg": 15, "protein_g": 0.4, "calories": 12},
        "Zucchini (summer squash), cooked": {"weight_g": 45, "phe_mg": 15, "protein_g": 0.4, "calories": 9},
    },
    "Fruits": {
        "Banana, sliced": {"weight_g": 42, "phe_mg": 16, "protein_g": 0.4, "calories": 39},
    }
}

def calculate_age_months(birth_year, birth_month, birth_day):
    """Calculate age in months from birth date"""
    today = datetime.today()
    return (today.year - birth_year) * 12 + (today.month - birth_month)

def get_phe_deletion_protocol(phe_mg_dl):
    """Determine PHE deletion hours based on diagnostic PHE level"""
    if phe_mg_dl < 4:
        return 0, "No deletion needed - PHE level is acceptable"
    elif 4 <= phe_mg_dl < 10:
        return 24, "Monitor daily, add PHE when level reaches 5 mg/dL"
    elif 10 <= phe_mg_dl < 20:
        return 48, "Monitor daily, add PHE when level reaches 5 mg/dL"
    elif 20 <= phe_mg_dl < 40:
        return 72, "Monitor daily, add PHE when level reaches 5 mg/dL"
    else:
        return 96, "Monitor daily, add PHE when level reaches 5 mg/dL"

def get_initial_phe_dose(diagnostic_phe_mg_dl, weight_kg):
    """Calculate initial PHE mg/kg after deletion period"""
    if diagnostic_phe_mg_dl <= 10:
        mg_per_kg = 70
    elif 10 < diagnostic_phe_mg_dl <= 20:
        mg_per_kg = 55
    elif 20 < diagnostic_phe_mg_dl <= 30:
        mg_per_kg = 45
    elif 30 < diagnostic_phe_mg_dl <= 40:
        mg_per_kg = 35
    else:
        mg_per_kg = 25
    return mg_per_kg * weight_kg, mg_per_kg

def get_infant_daily_needs(age_months, weight_kg):
    """Get daily nutritional needs for infants 0-12 months"""
    needs = {}
    
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
    else:
        needs['protein_g_per_kg'] = 3.0
        needs['phe_mg_per_kg_min'] = 10
        needs['phe_mg_per_kg_max'] = 35
        needs['energy_kcal_per_kg'] = 105
        needs['fluid_ml_per_kg'] = 135
        needs['age_group'] = '9-12 months'
    
    needs['protein_g'] = needs['protein_g_per_kg'] * weight_kg
    needs['phe_mg_min'] = needs['phe_mg_per_kg_min'] * weight_kg
    needs['phe_mg_max'] = needs['phe_mg_per_kg_max'] * weight_kg
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    needs['energy_kcal'] = needs['energy_kcal_per_kg'] * weight_kg
    needs['fluid_ml'] = needs['fluid_ml_per_kg'] * weight_kg
    
    return needs

def get_child_adult_daily_needs(age_months, weight_kg, sex):
    """Get daily nutritional needs for children and adults"""
    needs = {}
    
    if age_months < 48:
        needs['phe_mg_min'] = 200
        needs['phe_mg_max'] = 400
        needs['protein_g'] = 30
        needs['energy_kcal'] = 1300
        needs['age_group'] = '1-4 years'
    elif age_months < 84:
        needs['phe_mg_min'] = 210
        needs['phe_mg_max'] = 450
        needs['protein_g'] = 35
        needs['energy_kcal'] = 1700
        needs['age_group'] = '4-7 years'
    elif age_months < 132:
        needs['phe_mg_min'] = 220
        needs['phe_mg_max'] = 500
        needs['protein_g'] = 40
        needs['energy_kcal'] = 2400
        needs['age_group'] = '7-11 years'
    elif age_months < 180:
        if sex == "Female":
            needs['phe_mg_min'] = 250
            needs['phe_mg_max'] = 750
            needs['protein_g'] = 50
            needs['energy_kcal'] = 2200
        else:
            needs['phe_mg_min'] = 225
            needs['phe_mg_max'] = 900
            needs['protein_g'] = 55
            needs['energy_kcal'] = 2700
        needs['age_group'] = '11-15 years'
    elif age_months < 228:
        if sex == "Female":
            needs['phe_mg_min'] = 230
            needs['phe_mg_max'] = 700
            needs['protein_g'] = 55
            needs['energy_kcal'] = 2100
        else:
            needs['phe_mg_min'] = 295
            needs['phe_mg_max'] = 1100
            needs['protein_g'] = 65
            needs['energy_kcal'] = 2800
        needs['age_group'] = '15-19 years'
    else:
        if sex == "Female":
            needs['phe_mg_min'] = 220
            needs['phe_mg_max'] = 700
            needs['protein_g'] = 60
            needs['energy_kcal'] = 2100
        else:
            needs['phe_mg_min'] = 290
            needs['phe_mg_max'] = 1200
            needs['protein_g'] = 70
            needs['energy_kcal'] = 2900
        needs['age_group'] = '19+ years'
    
    needs['phe_mg_target'] = (needs['phe_mg_min'] + needs['phe_mg_max']) / 2
    return needs

def calculate_phenex_amount(protein_needed_g, protein_from_food_g, age_months):
    """Calculate grams of Phenex powder needed"""
    protein_from_phenex = protein_needed_g - protein_from_food_g
    
    if age_months < 24:
        phenex_grams = (protein_from_phenex * 100) / 15
        phenex_type = "Phenex-1"
        protein_per_100g = 15
        calories_per_100g = 480
    else:
        phenex_grams = (protein_from_phenex * 100) / 30
        phenex_type = "Phenex-2"
        protein_per_100g = 30
        calories_per_100g = 410
    
    actual_protein = (phenex_grams / 100) * protein_per_100g
    calories = (phenex_grams / 100) * calories_per_100g
    
    return {
        'phenex_type': phenex_type,
        'phenex_grams': phenex_grams,
        'protein_g': actual_protein,
        'calories_kcal': calories
    }

def calculate_milk_amount(phe_target_mg, milk_type):
    """Calculate milk amount based on PHE target"""
    if milk_type == "Breast Milk (Human Milk)":
        phe_per_100ml = 48
        protein_per_100ml = 1.07
        energy_per_100ml = 72
    else:
        phe_per_100ml = 59
        protein_per_100ml = 1.40
        energy_per_100ml = 68
    
    milk_ml = (phe_target_mg / phe_per_100ml) * 100
    
    return {
        'milk_type': milk_type,
        'milk_ml': milk_ml,
        'phe_mg': phe_target_mg,
        'protein_g': (milk_ml / 100) * protein_per_100ml,
        'calories_kcal': (milk_ml / 100) * energy_per_100ml
    }

def calculate_baby_diet_with_solids(age_months, weight_kg, current_phe_mg_dl, milk_type, solid_foods, age_hours=None):
    """Complete diet calculation for babies with solid foods"""
    result = {}
    
    needs = get_infant_daily_needs(age_months, weight_kg)
    result['needs'] = needs
    
    # Calculate total from solid foods
    total_solid_phe = sum(food['phe_mg'] for food in solid_foods)
    total_solid_protein = sum(food['protein_g'] for food in solid_foods)
    total_solid_calories = sum(food['calories'] for food in solid_foods)
    
    result['solid_foods'] = {
        'foods': solid_foods,
        'total_phe_mg': total_solid_phe,
        'total_protein_g': total_solid_protein,
        'total_calories': total_solid_calories
    }
    
    # Check deletion protocol
    result['deletion_needed'] = False
    if age_hours is not None and age_hours <= 96 and current_phe_mg_dl >= 4:
        deletion_hours, deletion_note = get_phe_deletion_protocol(current_phe_mg_dl)
        result['deletion_needed'] = True
        result['deletion_hours'] = deletion_hours
        result['deletion_note'] = deletion_note
        
        phenex_deletion = calculate_phenex_amount(needs['protein_g'], 0, age_months)
        result['deletion_phase'] = {
            'phenex': phenex_deletion,
            'water_ml': needs['fluid_ml'],
            'feedings_per_day': '6-8' if age_months < 6 else '4-6'
        }
        
        initial_phe_mg, phe_mg_per_kg = get_initial_phe_dose(current_phe_mg_dl, weight_kg)
        result['initial_phe_mg_per_kg'] = phe_mg_per_kg
        result['initial_phe_mg'] = initial_phe_mg
        phe_target = initial_phe_mg
    else:
        # Use target range for maintenance
        phe_target = needs['phe_mg_target']
        result['initial_phe_mg'] = phe_target
    
    # Calculate remaining PHE needed from milk
    remaining_phe = phe_target - total_solid_phe
    if remaining_phe < 0:
        remaining_phe = 0
    
    milk = calculate_milk_amount(remaining_phe, milk_type)
    result['milk'] = milk
    
    # Calculate Phenex
    total_protein_from_food_and_milk = total_solid_protein + milk['protein_g']
    phenex = calculate_phenex_amount(needs['protein_g'], total_protein_from_food_and_milk, age_months)
    result['phenex'] = phenex
    
    # Calculate water
    total_volume_needed = needs['fluid_ml']
    water_ml = max(0, total_volume_needed - milk['milk_ml'])
    result['water_ml'] = water_ml
    result['total_volume_ml'] = total_volume_needed
    
    result['feedings_per_day'] = '6-8' if age_months < 6 else '4-6'
    avg_feedings = 7 if age_months < 6 else 5
    result['ml_per_feeding'] = (milk['milk_ml'] + water_ml) / avg_feedings
    
    result['totals'] = {
        'protein_g': total_protein_from_food_and_milk + phenex['protein_g'],
        'phe_mg': total_solid_phe + milk['phe_mg'],
        'calories_kcal': total_solid_calories + milk['calories_kcal'] + phenex['calories_kcal']
    }
    
    return result

def format_age(birth_date):
    today = date.today()
    days_old = (today - birth_date).days
    months = days_old // 30
    years = months // 12

    if days_old < 30:
        return f"{days_old} days"
    elif months < 12:
        return f"{months} months"
    else:
        return f"{years} years {months % 12} months"

def display_baby_diet_plan_with_solids(result, weight_kg):
    """Display formatted baby diet plan with solid foods"""
    
    st.markdown("---")
    st.header("📋 Daily Diet Plan")
    
    needs = result['needs']
    st.subheader(f"Nutritional Targets ({needs['age_group']})")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Protein", f"{needs['protein_g']:.1f} g")
    with col2:
        st.metric("PHE Range", f"{needs['phe_mg_min']:.0f}-{needs['phe_mg_max']:.0f} mg")
    with col3:
        st.metric("Calories", f"{needs['energy_kcal']:.0f} kcal")
    
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
        st.success(f"**PHASE 2: ADDING PHE BACK**")
    
    # Display solid foods
    if result['solid_foods']['foods']:
        st.markdown("### 🍽️ Solid Foods (Beikost)")
        
        for i, food in enumerate(result['solid_foods']['foods']):
            with st.expander(f"**{food['meal']}:** {food['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Amount:** {food['weight_g']:.0f} g")
                    st.write(f"**PHE:** {food['phe_mg']:.0f} mg")
                with col2:
                    st.write(f"**Protein:** {food['protein_g']:.1f} g")
                    st.write(f"**Calories:** {food['calories']:.0f} kcal")
        
        st.markdown(f"""
        **Total from Solid Foods:**
        - PHE: {result['solid_foods']['total_phe_mg']:.0f} mg
        - Protein: {result['solid_foods']['total_protein_g']:.1f} g  
        - Calories: {result['solid_foods']['total_calories']:.0f} kcal
        """)
        
        st.markdown("---")
    
    st.markdown("### Daily Milk + Phenex Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {result['milk']['milk_type']}")
        st.markdown(f"""
        - **{result['milk']['milk_ml']:.0f} mL** per day
        - Provides:
          - PHE: {result['milk']['phe_mg']:.0f} mg
          - Protein: {result['milk']['protein_g']:.1f} g
          - Calories: {result['milk']['calories_kcal']:.0f} kcal
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
    
    st.markdown("#### Daily Schedule Example")
    st.info(f"""
    **Morning:** Phenex shake ({result['phenex']['phenex_grams']/3:.1f}g powder + {result['milk']['milk_ml']/3:.0f}mL milk + {result['water_ml']/3:.0f}mL water)
    
    **Lunch:** Solid foods + Phenex shake  
    
    **Dinner:** Solid foods + Phenex shake
    
    **Total Phenex:** {result['phenex']['phenex_grams']:.1f}g  
    **Total Milk:** {result['milk']['milk_ml']:.0f}mL  
    **Total Water:** {result['water_ml']:.0f}mL
    """)
    
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

def main():
    st.set_page_config(page_title="PKU Diet Manager", layout="wide")
    
    if 'profile_created' not in st.session_state:
        st.session_state.profile_created = False
    if 'solid_foods_list' not in st.session_state:
        st.session_state.solid_foods_list = []
    
    if not st.session_state.profile_created:
        st.title("PKU Diet Management System")
        st.markdown("""
        ### Welcome to your personalized PKU diet planner!
        
        This application helps manage phenylketonuria (PKU) diet by calculating:
        - Daily phenylalanine (PHE) targets
        - Phenex medical food requirements  
        - Milk amounts for infants (breast milk or formula)
        - Solid food planning for babies 6+ months
        - Meal planning for children and adults
        
        **Let's start by creating your profile.**
        """)
        
        st.markdown("---")
        st.header("Create Profile")
        
        age_category = st.radio(
            "Who is this profile for?",
            ["Baby (0-12 months)", "Child (1-12 years)", "Adult (12+ years)"]
        )
        
        st.markdown("---")
        
        if age_category == "Baby (0-12 months)":
            age_hours = st.number_input(
                'Age in hours (if newborn ≤96 hours):',
                min_value=0,
                max_value=2400,
                value=0,
                help="Enter age in hours if baby is ≤96 hours old for PHE deletion protocol"
            )
            
            st.markdown("**Milk Type Selection:**")
            milk_type = st.radio(
                "What type of milk will baby receive?",
                ["Breast Milk (Human Milk)", "Similac With Iron"],
                help="From Table 1-2, p.12: Human Milk has 48 mg PHE/100mL; Similac has 59 mg PHE/100mL"
            )
        else:
            age_hours = None
            milk_type = None
            
        if age_category != "Baby (0-12 months)":
            sex = st.radio("Sex:", ["Male", "Female"])
        else:
            sex = "Male"
        
        col1, col2 = st.columns(2)
        
        with col1:
            units = st.radio("Units:", ["Metric", "Imperial"])
            
            if units == "Metric":
                weight = st.number_input('Weight (kg):', min_value=0.0, step=0.1, key='weight_kg')
                height_cm = st.number_input('Height (cm):', min_value=0.0, step=1.0, key='height_cm')
                height = height_cm / 100
            else:
                weight_lbs = st.number_input('Weight (lbs):', min_value=0.0, step=0.1, key='weight_lbs')
                weight = weight_lbs * 0.453592
                height_in = st.number_input('Height (inches):', min_value=0.0, step=1.0, key='height_in')
                height = height_in * 0.0254
                height_cm = height_in * 2.54

        with col2:
            st.markdown("**Birth Date:**")
            birth_year = st.number_input('Year:', min_value=1900, max_value=datetime.now().year, value=2024, key='birth_year_input')
            birth_month = st.number_input('Month:', min_value=1, max_value=12, value=1, key='birth_month_input')
            birth_day = st.number_input('Day:', min_value=1, max_value=31, value=1, key='birth_day_input')
            
            current_phe = st.number_input(
                'Current Blood PHE Level (mg/dL):',
                min_value=0.0,
                step=0.1,
                help="Enter your most recent blood phenylalanine level",
                key='current_phe_input'
            )

        if st.button("Calculate Diet Plan", type="primary"):
            if weight == 0 or height == 0:
                st.error("Weight and height must be greater than 0.")
            elif current_phe is None or current_phe == 0:
                st.error("Please enter a current PHE level.")
            else:
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
                st.session_state.user_age_hours = age_hours
                st.session_state.user_milk_type = milk_type if age_category == "Baby (0-12 months)" else None
                st.rerun()
    
    else:
        st.sidebar.header("👤 Profile")
        
        birth_date = date(
            int(st.session_state.user_birth_year),
            int(st.session_state.user_birth_month),
            int(st.session_state.user_birth_day)
        )
        
        age_display = format_age(birth_date)
        st.sidebar.write(f"**Age:** {age_display}")
        st.sidebar.write(f"**Weight:** {st.session_state.user_weight:.1f} kg")
        st.sidebar.write(f"**Height:** {st.session_state.user_height_cm:.1f} cm")
        st.sidebar.write(f"**Current PHE:** {st.session_state.user_current_phe:.1f} mg/dL")
        
        if st.sidebar.button("🔄 Create New Profile"):
            st.session_state.profile_created = False
            st.session_state.solid_foods_list = []
            st.rerun()
        
        st.sidebar.markdown("---")
        
        age_months = calculate_age_months(
            st.session_state.user_birth_year,
            st.session_state.user_birth_month,
            st.session_state.user_birth_day
        )
        
        st.title("PKU Diet Plan")
        
        if st.session_state.user_age_category == "Baby (0-12 months)":
            # Check if baby is old enough for solid foods
            if age_months >= 6:
                st.info("💡 Baby is 6+ months old and can have solid foods (beikost)!")
                
                with st.expander("🍽️ Add Solid Foods to Diet Plan", expanded=True):
                    st.markdown("Select foods from the PKU protocol database (Table 1-3)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        meal_type = st.selectbox(
                            "Meal:",
                            ["Lunch", "Dinner", "Breakfast", "Snack"],
                            key='meal_select'
                        )
                        
                        # Combine baby foods and table foods for older babies
                        all_foods = {}
                        if age_months < 9:
                            all_foods = BABY_FOODS.copy()
                        else:
                            all_foods = {**BABY_FOODS, **TABLE_FOODS}
                        
                        category = st.selectbox(
                            "Food Category:",
                            list(all_foods.keys()),
                            key='category_select'
                        )
                        
                        food_name = st.selectbox(
                            "Food:",
                            list(all_foods[category].keys()),
                            key='food_select'
                        )
                    
                    with col2:
                        food_data = all_foods[category][food_name]
                        
                        st.markdown(f"**Standard Serving:**")
                        st.write(f"Weight: {food_data['weight_g']}g")
                        st.write(f"PHE: {food_data['phe_mg']}mg")
                        st.write(f"Protein: {food_data['protein_g']}g")
                        st.write(f"Calories: {food_data['calories']} kcal")
                        
                        servings = st.number_input(
                            "Number of servings:",
                            min_value=0.5,
                            max_value=3.0,
                            value=1.0,
                            step=0.5,
                            key='servings_input'
                        )
                    
                    if st.button("➕ Add Food", key='add_food_btn'):
                        food_entry = {
                            'meal': meal_type,
                            'name': food_name,
                            'weight_g': food_data['weight_g'] * servings,
                            'phe_mg': food_data['phe_mg'] * servings,
                            'protein_g': food_data['protein_g'] * servings,
                            'calories': food_data['calories'] * servings,
                            'servings': servings
                        }
                        st.session_state.solid_foods_list.append(food_entry)
                        st.success(f"Added {servings} serving(s) of {food_name}!")
                        st.rerun()
                    
                    if st.session_state.solid_foods_list:
                        st.markdown("---")
                        st.markdown("### Current Solid Foods:")
                        for i, food in enumerate(st.session_state.solid_foods_list):
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.write(f"**{food['meal']}:** {food['name']}")
                            with col2:
                                st.write(f"{food['weight_g']:.0f}g | {food['phe_mg']:.0f}mg PHE")
                            with col3:
                                if st.button("🗑️", key=f"del_{i}"):
                                    st.session_state.solid_foods_list.pop(i)
                                    st.rerun()
                        
                        if st.button("🗑️ Clear All Foods"):
                            st.session_state.solid_foods_list = []
                            st.rerun()
            
            # Calculate diet
            result = calculate_baby_diet_with_solids(
                age_months,
                st.session_state.user_weight,
                st.session_state.user_current_phe,
                st.session_state.user_milk_type,
                st.session_state.solid_foods_list,
                st.session_state.user_age_hours
            )
            result['current_phe_mg_dl'] = st.session_state.user_current_phe
            display_baby_diet_plan_with_solids(result, st.session_state.user_weight)
            
        else:
            needs = get_child_adult_daily_needs(
                age_months,
                st.session_state.user_weight,
                st.session_state.user_sex
            )
            display_child_adult_plan(needs, age_months, st.session_state.user_weight)
        
        st.markdown("---")
        st.header("📖 Important Information")
        
        with st.expander("Understanding Your Numbers"):
            st.markdown("""
            **Blood PHE Levels (Section VI, p.2):**
            - **Target for children:** 2-5 mg/dL (120-300 µmol/L)
            - **Target for adults:** 2-10 mg/dL (120-600 µmol/L)
            - Levels should be checked regularly (weekly to monthly depending on age and control)
            
            **What affects PHE levels:**
            - Amount of PHE eaten from foods
            - Adequate calorie and protein intake
            - Illness or stress
            - Growth spurts in children
            """)
        
        with st.expander("Introducing Solid Foods (Section VIII.A.3, p.5)"):
            st.markdown("""
            **When to Start:**
            - Begin beikost (baby foods) at 3-4 months or when developmentally ready
            - Gradually displace PHE from milk/formula with solid foods
            
            **How to Progress:**
            - Start with low-PHE vegetables and fruits
            - Measure all foods with a gram scale for accuracy
            - Keep records of all foods eaten
            - Adjust milk amount as solid food intake increases
            
            **Important:**
            - Always maintain total PHE within target range
            - Continue Phenex-1 for protein needs
            - Monitor blood PHE twice weekly when introducing new foods
            """)
        
        with st.expander("⚠️ When to Contact Your Metabolic Clinic"):
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
        
        st.markdown("---")
        st.warning("""
        **Important Disclaimer:**
        - This app is a tool to help manage your PKU diet
        - Always follow your metabolic team's specific recommendations
        - Never make major diet changes without consulting your doctor/dietitian
        - Blood PHE monitoring is essential - never skip tests
        - PKU is manageable with proper diet and monitoring
        """)

if __name__ == "__main__":
    main()