import streamlit as st
import pandas as pd
import random
import datetime

# Load food CSV
food_data = pd.read_csv("Mexican_Foods.csv")
food_names = food_data['Meal'].dropna().unique().tolist()

# CSV file to save responses
DATA_FILE = "user_responses.csv"

# Function to generate random PHE tolerance based on age
def generate_phe(age):
    if age <= 18:  # children
        return random.randint(120, 360)
    else:  # adults
        return random.randint(120, 600)

# ------------------- Initialize session state ------------------- #
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'food_index' not in st.session_state:
    st.session_state.food_index = 0
if 'selected_foods' not in st.session_state:
    st.session_state.selected_foods = random.sample(list(food_names), 10)

# ------------------ PAGE 0: Consent ------------------ #
if st.session_state.page == 0:
    st.title("PKU Dietary Preference Study")
    st.markdown("""
    **About the research:**  
    Phenylketonuria (PKU) is a rare metabolic disorder caused by variants in the PAH gene, leading to elevated phenylalanine (PHE) levels in the blood.  

    This study collects dietary preferences and basic patient information to develop a personalized food recommendation system for PKU management.

    **Your data:**  
    - Height, weight, age, gender  
    - Food ratings  
    - PHE tolerance  

    Your responses are stored **locally** in a CSV file. No personal identifiers are shared externally.

    By proceeding, you consent to participate in this research.
    """)
    if st.button("I Consent"):
        st.session_state.page = 1

# ------------------ PAGE 1: User Info ------------------ #
elif st.session_state.page == 1:
    st.title("User Information")
    st.write("Please provide the following information:")

    name = st.text_input("Full Name")
    email = st.text_input("Email")

    age = st.number_input("Age (years)", min_value=1, max_value=120, value=18)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Height input
    height_unit = st.radio("Height Unit", ["cm", "ft/in"])
    if height_unit == "cm":
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    else:
        ft = st.number_input("Height (ft)", min_value=1, max_value=8, value=5)
        inch = st.number_input("Height (in)", min_value=0, max_value=11, value=7)
        height = ft*30.48 + inch*2.54

    # Weight input
    weight_unit = st.radio("Weight Unit", ["kg", "lbs"])
    if weight_unit == "kg":
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    else:
        lbs = st.number_input("Weight (lbs)", min_value=20, max_value=700, value=150)
        weight = lbs*0.453592

    # Generate PHE
    if st.button("Next"):
        phe = generate_phe(age)
        st.session_state.user_data.update({
            "Name": name,
            "Email": email,
            "Age": age,
            "Gender": gender,
            "Height_cm": round(height, 1),
            "Weight_kg": round(weight, 1),
            "PHE_tolerance": phe
        })
        st.session_state.page = 2

    if 'phe' not in st.session_state:
        st.session_state.phe = generate_phe(age)
    st.markdown(f"**Your estimated PHE tolerance:** {st.session_state.phe} μmol/L")

# ------------------ PAGE 2: Food Ratings ------------------ #
elif st.session_state.page == 2:
    st.title("Food Rating")
    idx = st.session_state.food_index
    food = st.session_state.selected_foods[idx]

    st.write(f"Food {idx+1}/10: **{food}**")
    rating = st.slider("Rate this food", 0, 5, 1, key=f"slider_{idx}")
    st.session_state.user_data[food] = rating

    # Navigation buttons
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Previous") and idx > 0:
            st.session_state.food_index -= 1
    with col2:
        if st.button("Next") and idx < 9:
            st.session_state.food_index += 1
    with col3:
        if st.button("Submit") and idx == 9:
            st.session_state.user_data["Timestamp"] = datetime.datetime.now().isoformat()
            df_new = pd.DataFrame([st.session_state.user_data])
            try:
                df_existing = pd.read_csv(DATA_FILE)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except FileNotFoundError:
                df_combined = df_new
            df_combined.to_csv(DATA_FILE, index=False)
            st.success("Your responses have been saved. Thank you for participating!")