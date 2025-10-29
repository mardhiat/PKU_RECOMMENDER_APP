import streamlit as st
import pandas as pd
import random
import datetime

# Load all cuisine CSV files
CUISINE_FILES = {
    "African Foods": "African_Foods.csv",
    "Central European Foods": "Central_European_Foods.csv",
    "Chinese Foods": "Chinese_Foods.csv",
    "Eastern European Foods": "Eastern_European_Foods.csv",
    "Indian Foods": "Indian_Foods.csv",
    "Italian Foods": "Italian_Foods.csv",
    "Japanese Foods": "Japanese_Foods.csv",
    "Mediterranean Foods": "Mediterranean_Foods.csv",
    "Mexican Foods": "Mexican_Foods.csv",
    "Scottish-Irish Foods": "Scottish-Irish_Foods.csv"
}

# CSV file to save responses
DATA_FILE = "user_responses.csv"

# Function to generate random PHE tolerance based on age
def generate_phe(age):
    if age <= 18:  # children
        return random.randint(120, 360)
    else:  # adults
        return random.randint(120, 600)

# Function to load cuisine data and get meals with ingredients
def load_cuisine_meals(cuisine_name):
    """Load meals from a cuisine CSV and return dict of {meal: [ingredients]}"""
    try:
        df = pd.read_csv(CUISINE_FILES[cuisine_name])
        meals = {}
        current_meal = None
        
        for _, row in df.iterrows():
            meal_name = row.get('Meal', None)
            ingredient = row.get('Ingredient', None)
            
            # If we find a meal name, start a new meal entry
            if pd.notna(meal_name):
                current_meal = meal_name
                meals[current_meal] = []
            
            # Add ingredient to current meal
            if pd.notna(ingredient) and current_meal:
                meals[current_meal].append(ingredient)
        
        return meals
    except Exception as e:
        st.error(f"Error loading {cuisine_name}: {e}")
        return {}

#  Initialize session state  #
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'food_index' not in st.session_state:
    st.session_state.food_index = 0
if 'selected_cuisines' not in st.session_state:
    st.session_state.selected_cuisines = []
if 'selected_foods' not in st.session_state:
    st.session_state.selected_foods = []
if 'food_ingredients' not in st.session_state:
    st.session_state.food_ingredients = {}

#  PAGE 0: Consent 
if st.session_state.page == 0:
    st.title("PKU Dietary Preference Study")
    st.markdown("""
    **About the research:**  
    Phenylketonuria (PKU) is a rare metabolic disorder caused by variants in the PAH gene, leading to elevated phenylalanine (PHE) levels in the blood.  

    This study collects dietary preferences and basic patient information to develop a personalized food recommendation system for PKU management.

    **Your data:**  
    - Height, weight, age, gender  
    - Food ratings from your chosen cuisines
    - PHE tolerance  

    Your responses are stored **safely**. No personal identifiers are shared externally.

    By proceeding, you consent to participate in this research.
    """)
    if st.button("I Consent"):
        st.session_state.page = 1
        st.rerun()

#  PAGE 1: User Info 
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
    phe = generate_phe(age)
    st.info(f"**Your estimated PHE tolerance:** {phe} μmol/L")

    # Next button
    if st.button("Next"):
        if not name or not email:
            st.error("Please fill in your name and email.")
        else:
            st.session_state.user_data.update({
                "Name": name,
                "Email": email,
                "Age": age,
                "Gender": gender,
                "Height_cm": round(height, 1),
                "Weight_kg": round(weight, 1),
                "PHE_tolerance": phe
            })
            st.session_state.page = 1.5
            st.rerun()

#  PAGE 1.5: Cuisine Selection 
elif st.session_state.page == 1.5:
    st.title("Select Your Preferred Cuisines")
    st.write("Choose one or more cuisines you'd like to rate foods from:")
    
    selected = st.multiselect(
        "Select cuisines (you can choose multiple):",
        options=list(CUISINE_FILES.keys()),
        default=[]
    )
    
    if st.button("Continue to Food Rating"):
        if not selected:
            st.error("Please select at least one cuisine.")
        else:
            st.session_state.selected_cuisines = selected
            
            # Load 15 random foods from each selected cuisine
            all_foods = []
            food_ingredients = {}
            
            for cuisine in selected:
                meals_dict = load_cuisine_meals(cuisine)
                if meals_dict:
                    # Get all meal names from this cuisine
                    meal_names = list(meals_dict.keys())
                    # Sample up to 15 random meals
                    sampled_meals = random.sample(meal_names, min(15, len(meal_names)))
                    
                    # Add to our list with cuisine prefix
                    for meal in sampled_meals:
                        food_key = f"{meal} ({cuisine})"
                        all_foods.append(food_key)
                        food_ingredients[food_key] = meals_dict[meal]
            
            st.session_state.selected_foods = all_foods
            st.session_state.food_ingredients = food_ingredients
            st.session_state.user_data["Selected_Cuisines"] = ", ".join(selected)
            st.session_state.page = 2
            st.rerun()

#  PAGE 2: Food Ratings 
elif st.session_state.page == 2:
    st.title("Food Rating")
    
    total_foods = len(st.session_state.selected_foods)
    idx = st.session_state.food_index
    
    if idx >= total_foods:
        st.session_state.page = 3
        st.rerun()
    
    food = st.session_state.selected_foods[idx]
    ingredients = st.session_state.food_ingredients.get(food, [])
    
    st.write(f"**Food {idx+1}/{total_foods}**")
    st.subheader(food)
    
    # Display ingredients
    if ingredients:
        st.write("**Ingredients:**")
        st.write(", ".join(ingredients))
    else:
        st.write("*No ingredient information available*")
    
    # Rating slider
    rating = st.slider(
        "How much did you enjoy this food?", 
        0, 5, 3, 
        key=f"slider_{idx}",
        help="0 = Never tried it before, 5 = Loved it!"
    )
    
    # Store rating
    st.session_state.user_data[food] = rating
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        if st.button("Previous") and idx > 0:
            st.session_state.food_index -= 1
            st.rerun()
    
    with col2:
        st.write(f"Progress: {idx+1}/{total_foods}")
    
    with col3:
        if idx < total_foods - 1:
            if st.button("Next"):
                st.session_state.food_index += 1
                st.rerun()
        else:
            if st.button("Submit"):
                st.session_state.page = 3
                st.rerun()

#  PAGE 3: Thank You 
elif st.session_state.page == 3:
    st.title("Thank You!")
    
    # Add timestamp
    st.session_state.user_data["Timestamp"] = datetime.datetime.now().isoformat()
    
    try:
        # Import Google Sheets libraries
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        
        # Connect to Google Sheets
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["gcp_service_account"], 
            scope
        )
        client = gspread.authorize(creds)
        
        # Open the sheet
        sheet = client.open("ratingsappdata").sheet1
        
        # Combine existing headers with any new keys
        existing_headers = sheet.row_values(1)
        all_keys = list(st.session_state.user_data.keys())

        # Add any new keys to the sheet header
        for key in all_keys:
            if key not in existing_headers:
                existing_headers.append(key)

        # If sheet is empty, add headers
        if not existing_headers:
            sheet.append_row(all_keys)
        else:
            sheet.update('A1', [existing_headers])

        
        # Prepare data row matching header order
        headers = sheet.row_values(1)
        row_data = []
        for header in headers:
            value = st.session_state.user_data.get(header, "")
            row_data.append(str(value))
        
        # Append data
        sheet.append_row(row_data)
        
        st.success(" Your responses have been saved successfully!")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        
        # Fallback: save to CSV locally (for backup/testing)
        df_new = pd.DataFrame([st.session_state.user_data])
        try:
            df_existing = pd.read_csv(DATA_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_combined = df_new
        df_combined.to_csv(DATA_FILE, index=False)
        st.warning("Data saved to local CSV as backup.")
    
    st.markdown("""
    ### Summary of Your Participation:
    - **Foods Rated:** {} foods from {} cuisine(s)
    - **Your PHE Tolerance:** {} μmol/L
    
    Thank you for contributing to PKU dietary research!
    
    Your responses will help develop better food recommendation systems for PKU patients.
    """.format(
        len(st.session_state.selected_foods),
        len(st.session_state.selected_cuisines),
        st.session_state.user_data.get("PHE_tolerance", "N/A")
    ))
    

    # --- FOOD RECOMMENDATION SECTION --- #
    st.subheader("Recommended Foods for You 🍽️")

    # Get rated foods
    rated_foods = {food: rating for food, rating in st.session_state.user_data.items() if isinstance(rating, (int, float))}
    liked_foods = [food for food, rating in rated_foods.items() if rating >= 4]

    # Build full list of all foods from selected cuisines
    all_foods = {}
    for cuisine in st.session_state.selected_cuisines:
        meals_dict = load_cuisine_meals(cuisine)
        for meal, ingredients in meals_dict.items():
            food_key = f"{meal} ({cuisine})"
            all_foods[food_key] = ingredients

    # Find foods the user didn’t rate
    unrated_foods = [f for f in all_foods.keys() if f not in rated_foods]

    # Score foods by ingredient similarity to liked foods
    food_scores = {}
    for food in unrated_foods:
        ingredients = set(all_foods[food])
        similarity_score = 0
        for liked in liked_foods:
            liked_ing = set(all_foods.get(liked, []))
            intersection = ingredients.intersection(liked_ing)
            union = ingredients.union(liked_ing)
            if union:
                similarity_score += len(intersection) / len(union)
        food_scores[food] = similarity_score

    # Sort and show top 5 recommendations
    top_5 = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    if top_5:
        for food, score in top_5:
            st.markdown(f"**{food}** (Similarity: {score:.2f})")
            st.write(", ".join(all_foods.get(food, [])))
            st.divider()
    else:
        st.info("Not enough data yet to generate recommendations.")



    if st.button("Start New Response"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()