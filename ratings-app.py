import streamlit as st
import pandas as pd
import random
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# Function to generate random PHE tolerance based on age (in mg/day)
def generate_phe(age):
    """Generate PHE tolerance in mg/day (range 3-40)"""
    if age <= 18:  # children
        return random.randint(3, 25)
    else:  # adults
        return random.randint(10, 40)

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

# Load historical user data from Google Sheets
@st.cache_data(ttl=300)  # Cache for 5 minutes
def loading_user_data_from_sheets():
    """Load all historical user ratings from Google Sheets"""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["gcp_service_account"], 
            scope
        )
        client = gspread.authorize(creds)
        sheet = client.open("ratingsappdata").sheet1
        
        # Get all data
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        return df
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")
        return pd.DataFrame()

# Content-based filtering recommendation
def content_based_recommendation(all_foods, liked_foods, unrated_foods):
    """Content-based filtering: recommend based on ingredient similarity"""
    food_scores = {}
    for food in unrated_foods:
        ingredients = set(all_foods.get(food, []))
        similarity_score = 0
        for liked in liked_foods:
            liked_ing = set(all_foods.get(liked, []))
            intersection = ingredients.intersection(liked_ing)
            union = ingredients.union(liked_ing)
            if union:
                # Jaccard similarity
                similarity_score += len(intersection) / len(union)
        food_scores[food] = similarity_score / max(len(liked_foods), 1)
    return food_scores

# Collaborative filtering recommendation
def collaborative_filtering_recommendation(historical_df, current_user_ratings, unrated_foods):
    """Collaborative filtering: recommend based on similar users' preferences"""
    if historical_df.empty or len(historical_df) < 2:
        return {}
    
    # Extract only food rating columns (those with parentheses indicating cuisine)
    food_columns = [col for col in historical_df.columns if '(' in str(col) and ')' in str(col)]
    
    if not food_columns:
        return {}
    
    # Create user-item matrix
    ratings_matrix = historical_df[food_columns].copy()
    
    # Convert to numeric, replace empty/non-numeric with NaN
    for col in food_columns:
        ratings_matrix[col] = pd.to_numeric(ratings_matrix[col], errors='coerce')
    
    # Fill NaN with 0 for similarity calculation
    ratings_matrix_filled = ratings_matrix.fillna(0)
    
    # Create current user's rating vector
    current_user_vector = []
    for col in food_columns:
        if col in current_user_ratings:
            current_user_vector.append(current_user_ratings[col])
        else:
            current_user_vector.append(0)
    
    current_user_vector = np.array(current_user_vector).reshape(1, -1)
    
    # Calculate cosine similarity between current user and all historical users
    similarities = cosine_similarity(current_user_vector, ratings_matrix_filled.values)
    
    # Get top 5 similar users
    similar_users_idx = similarities[0].argsort()[-5:][::-1]
    
    # Predict ratings for unrated foods based on similar users
    food_scores = {}
    for food in unrated_foods:
        if food not in food_columns:
            continue
        
        weighted_sum = 0
        similarity_sum = 0
        
        for idx in similar_users_idx:
            user_rating = ratings_matrix.iloc[idx][food]
            if pd.notna(user_rating) and user_rating > 0:
                similarity_score = similarities[0][idx]
                weighted_sum += similarity_score * user_rating
                similarity_sum += similarity_score
        
        if similarity_sum > 0:
            food_scores[food] = weighted_sum / similarity_sum
        else:
            food_scores[food] = 0
    
    return food_scores

# Hybrid filtering recommendation
def hybrid_recommendation(content_scores, collaborative_scores, alpha=0.5):
    """Hybrid filtering: combine content-based and collaborative filtering"""
    hybrid_scores = {}
    
    # Normalize scores to 0-1 range
    def normalize_scores(scores):
        if not scores or max(scores.values()) == 0:
            return scores
        max_score = max(scores.values())
        return {k: v/max_score for k, v in scores.items()}
    
    content_norm = normalize_scores(content_scores)
    collab_norm = normalize_scores(collaborative_scores)
    
    # Combine all foods
    all_foods = set(content_norm.keys()).union(set(collab_norm.keys()))
    
    for food in all_foods:
        content_score = content_norm.get(food, 0)
        collab_score = collab_norm.get(food, 0)
        # Weighted combination
        hybrid_scores[food] = alpha * content_score + (1 - alpha) * collab_score
    
    return hybrid_scores

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
    ### About Phenylketonuria (PKU)
    
    Phenylketonuria (PKU) is a rare inherited metabolic disorder caused by variations in the PAH gene, 
    leading to elevated levels of phenylalanine (PHE) in the blood. Managing PHE levels through strict 
    dietary control is essential to prevent neurological complications and support healthy development.
    
    This application is designed to support individuals with PKU by collecting user ratings on foods 
    that are safe or restricted in the PKU diet.
    
    ### Important Information About PKU Diet
    
    People with PKU cannot safely consume about **90% of common foods worldwide**. Only about **5% of foods** 
    are considered safe for regular consumption, while another **5% are restricted** and must be eaten only 
    in carefully controlled amounts, depending on each individual's tolerance level. The other **90% of foods** 
    are strictly forbidden due to their high phenylalanine content.
    
    Therefore, the variety of foods available in this app is limited and primarily includes **fruits, 
    vegetables, and carbohydrate-based items** that are naturally lower in phenylalanine.
    
    **Please note that meat, chicken, fish, nuts, and dairy products are prohibited for individuals with PKU.** 
    Therefore, all dishes included in this app are selected to align with PKU dietary guidelines and ensure safety.
    
    **Your careful and thoughtful ratings are extremely valuable**—they will help improve PKU-friendly food 
    recommendations and make a meaningful contribution to the PKU Society and broader community research efforts.
    
    ---
    
    ### Purpose of the Study
    
    This study collects dietary preferences and basic participant information to develop a personalized 
    food recommendation system that supports PKU dietary management.
    
    **Data collected includes:**
    - Height, weight, age, and gender
    - Food ratings from your selected cuisines
    - Dietary tolerance level
    
    All data are stored securely, and no personal identifiers are shared externally. Participation is 
    voluntary, and you may withdraw at any time.
    
    ---
    
    ### Food Rating Guidelines
    
    In this application, you are asked to rate each food item on a scale from **0 to 5** based on your 
    personal preference and familiarity:
    
    - **0 – No Opinion / Not Sure:** You are unfamiliar with the food or unsure about it.
    - **1 – Strongly Dislike:** You do not like this food at all.
    - **2 – Dislike:** You generally would not choose this food.
    - **3 – Neutral:** You neither like nor dislike this food.
    - **4 – Like:** You enjoy this food and would choose it.
    - **5 – Strongly Like:** You really like this food and it is one of your favorites.
    
    Your ratings help the system learn your preferences and improve the accuracy of PKU-friendly food 
    recommendations tailored to your dietary needs.
    
    ---
    
    ### Feedback on Recommended Foods
    
    At the end of the rating process, the application will display **two sets of recommendations**:
    
    1. **Top 5 Dishes from Your Selected Cuisine(s)** – Based on the cuisine category you selected 
       (for example, Mediterranean, Asian, or European), the system will recommend your top five dishes 
       from within those selected cuisines.
    
    2. **Top 5 Dishes Across All Cuisines** – The system will also suggest the five most suitable dishes 
       overall, selected from all cuisine categories available in the application.
    
    We kindly ask you to provide quick feedback on these recommendations by selecting **"Thumbs Up 👍"** 
    if you agree or like the suggestion, or **"Thumbs Down 👎"** if you do not.
    
    Your feedback on both sets of recommendations will help us refine the algorithm, enhance the accuracy 
    of personalized PKU-friendly food suggestions, and contribute valuable insights to the PKU Society's 
    ongoing research efforts.
    
    ---
    
    ### Consent
    
    By proceeding, you consent to participate in this research and allow your anonymized data to be used 
    for scientific purposes.
    """)
    
    if st.button("I Consent to Participate"):
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

    # Generate PHE (in mg/day, range 3-40)
    phe = generate_phe(age)
    st.info(f"**Your estimated dietary tolerance level:** {phe} mg/day")

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
                "Dietary_tolerance_mg_per_day": phe
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
                    # Sample up to 20 random meals
                    sampled_meals = random.sample(meal_names, min(20, len(meal_names)))
                    
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
    
    # Rating slider with improved help text
    rating = st.slider(
        "How much did you enjoy this food?", 
        0, 5, 3, 
        key=f"slider_{idx}",
        help="0 = No opinion/Not sure | 1 = Strongly dislike | 2 = Dislike | 3 = Neutral | 4 = Like | 5 = Strongly like"
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

#  PAGE 3: Thank You & Recommendations
elif st.session_state.page == 3:
    st.title("Thank You!")
    
    # Add timestamp
    st.session_state.user_data["Timestamp"] = datetime.datetime.now().isoformat()
    
    # Load historical data from Google Sheets
    st.info("Loading historical user data for collaborative recommendations...")
    historical_df = loading_user_data_from_sheets()
    
    # Get rated foods
    rated_foods = {food: rating for food, rating in st.session_state.user_data.items() 
                   if isinstance(rating, (int, float)) and "(" in str(food)}
    liked_foods = [food for food, rating in rated_foods.items() if rating >= 4]
    
    # --- RECOMMENDATION SECTION --- #
    st.subheader("🍽️ Personalized Food Recommendations")
    
    # Build foods from SELECTED cuisines only
    selected_cuisine_foods = {}
    for cuisine in st.session_state.selected_cuisines:
        meals_dict = load_cuisine_meals(cuisine)
        for meal, ingredients in meals_dict.items():
            food_key = f"{meal} ({cuisine})"
            selected_cuisine_foods[food_key] = ingredients

    # Build foods from ALL cuisines in the database (for global recommendations)
    all_cuisine_foods = {}
    for cuisine in CUISINE_FILES.keys():  # This loops through ALL 10 cuisines
        meals_dict = load_cuisine_meals(cuisine)
        for meal, ingredients in meals_dict.items():
            food_key = f"{meal} ({cuisine})"
            all_cuisine_foods[food_key] = ingredients

    # Find unrated foods
    unrated_selected = [f for f in selected_cuisine_foods.keys() if f not in rated_foods]
    unrated_all = [f for f in all_cuisine_foods.keys() if f not in rated_foods]
    
    if liked_foods:
        # === RECOMMENDATIONS FROM SELECTED CUISINES ===
        st.markdown("### 📋 Top 5 Dishes from Your Selected Cuisines")
        st.write(f"Based on cuisines you chose: **{', '.join(st.session_state.selected_cuisines)}**")
        
        # Content-Based
        content_scores_selected = content_based_recommendation(
            selected_cuisine_foods, liked_foods, unrated_selected
        )
        
        # Collaborative
        collaborative_scores_selected = collaborative_filtering_recommendation(
            historical_df, rated_foods, unrated_selected
        )
        
        # Hybrid
        hybrid_scores_selected = hybrid_recommendation(
            content_scores_selected, collaborative_scores_selected, alpha=0.6
        )
        
        # Display all three types
        tab1, tab2, tab3 = st.tabs([
            "🔀 Hybrid (Recommended)",
            "🎯 Content-Based", 
            "👥 Collaborative"
        ])
        
        with tab1:
            st.write("**Method:** Combined content-based + collaborative (60% content, 40% collaborative)")
            top_5 = sorted(hybrid_scores_selected.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_5 and top_5[0][1] > 0:
                for i, (food, score) in enumerate(top_5, 1):
                    st.markdown(f"**{i}. {food}** (Score: {score:.3f})")
                    st.write("*Ingredients:* " + ", ".join(selected_cuisine_foods.get(food, [])))
                    feedback = st.radio(
                        "Do you like this recommendation?",
                        ("👍 Thumbs Up", "👎 Thumbs Down"),
                        key=f"hybrid_sel_{i}",
                        horizontal=True
                    )
                    st.session_state.user_data[f"Hybrid_Selected_{food}"] = feedback
                    st.divider()
            else:
                st.info("Not enough data for hybrid recommendations.")
        
        with tab2:
            st.write("**Method:** Ingredient similarity analysis")
            top_5 = sorted(content_scores_selected.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_5 and top_5[0][1] > 0:
                for i, (food, score) in enumerate(top_5, 1):
                    st.markdown(f"**{i}. {food}** (Score: {score:.3f})")
                    st.write("*Ingredients:* " + ", ".join(selected_cuisine_foods.get(food, [])))
                    feedback = st.radio(
                        "Do you like this recommendation?",
                        ("👍 Thumbs Up", "👎 Thumbs Down"),
                        key=f"cb_sel_{i}",
                        horizontal=True
                    )
                    st.session_state.user_data[f"CB_Selected_{food}"] = feedback
                    st.divider()
            else:
                st.info("Not enough data for content-based recommendations.")
        
        with tab3:
            st.write("**Method:** Similar users' preferences")
            if not historical_df.empty:
                top_5 = sorted(collaborative_scores_selected.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_5 and top_5[0][1] > 0:
                    for i, (food, score) in enumerate(top_5, 1):
                        st.markdown(f"**{i}. {food}** (Score: {score:.3f})")
                        st.write("*Ingredients:* " + ", ".join(selected_cuisine_foods.get(food, [])))
                        feedback = st.radio(
                            "Do you like this recommendation?",
                            ("👍 Thumbs Up", "👎 Thumbs Down"),
                            key=f"collab_sel_{i}",
                            horizontal=True
                        )
                        st.session_state.user_data[f"Collab_Selected_{food}"] = feedback
                        st.divider()
                else:
                    st.info("Not enough similar users found.")
            else:
                st.info("No historical data available yet. Collaborative filtering requires multiple users.")
        
        st.markdown("---")
        
        # === RECOMMENDATIONS FROM ALL CUISINES ===
        st.markdown("### 🌍 Top 5 Dishes Across All Cuisines")
        st.write("Expanding recommendations to include all available cuisines")
        
        # Filter to EXCLUDE selected cuisines (prioritize new cuisines)
        unrated_other_cuisines = [
            f for f in unrated_all 
            if not any(cuisine in f for cuisine in st.session_state.selected_cuisines)
        ]
        
        # If we have dishes from other cuisines, use those; otherwise fall back to all
        unrated_for_global = unrated_other_cuisines if unrated_other_cuisines else unrated_all
        
        # Content-Based
        content_scores_all = content_based_recommendation(
            all_cuisine_foods, liked_foods, unrated_for_global
        )
        
        # Collaborative
        collaborative_scores_all = collaborative_filtering_recommendation(
            historical_df, rated_foods, unrated_for_global
        )
        
        # Hybrid - use different weighting for exploration (more collaborative)
        hybrid_scores_all = hybrid_recommendation(
            content_scores_all, collaborative_scores_all, alpha=0.4
        )
        
        # Display all three types
        tab4, tab5, tab6 = st.tabs([
            "🔀 Hybrid (Recommended)",
            "🎯 Content-Based", 
            "👥 Collaborative"
        ])
        
        with tab4:
            st.write("**Method:** Combined content-based + collaborative (40% content, 60% collaborative)")
            
            # Get diversified recommendations - ensure variety across cuisines
            def get_diverse_recommendations(scores_dict, max_items=5, max_per_cuisine=2):
                """Select diverse recommendations with limited items per cuisine"""
                sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
                selected = []
                cuisine_count = {}
                
                for food, score in sorted_items:
                    if len(selected) >= max_items:
                        break
                    
                    # Extract cuisine name from food (format: "Dish Name (Cuisine)")
                    cuisine = food.split('(')[-1].strip(')')
                    
                    # Check if we haven't exceeded max for this cuisine
                    if cuisine_count.get(cuisine, 0) < max_per_cuisine:
                        selected.append((food, score))
                        cuisine_count[cuisine] = cuisine_count.get(cuisine, 0) + 1
                
                return selected
            
            top_5 = get_diverse_recommendations(hybrid_scores_all, max_items=5, max_per_cuisine=2)
            
            if top_5 and top_5[0][1] > 0:
                for i, (food, score) in enumerate(top_5, 1):
                    st.markdown(f"**{i}. {food}** (Score: {score:.3f})")
                    st.write("*Ingredients:* " + ", ".join(all_cuisine_foods.get(food, [])))
                    feedback = st.radio(
                        "Do you like this recommendation?",
                        ("👍 Thumbs Up", "👎 Thumbs Down"),
                        key=f"hybrid_all_{i}",
                        horizontal=True
                    )
                    st.session_state.user_data[f"Hybrid_All_{food}"] = feedback
                    st.divider()
            else:
                st.info("Not enough data for hybrid recommendations.")
        
        with tab5:
            st.write("**Method:** Ingredient similarity analysis")
            top_5 = sorted(content_scores_all.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_5 and top_5[0][1] > 0:
                for i, (food, score) in enumerate(top_5, 1):
                    st.markdown(f"**{i}. {food}** (Score: {score:.3f})")
                    st.write("*Ingredients:* " + ", ".join(all_cuisine_foods.get(food, [])))
                    feedback = st.radio(
                        "Do you like this recommendation?",
                        ("👍 Thumbs Up", "👎 Thumbs Down"),
                        key=f"cb_all_{i}",
                        horizontal=True
                    )
                    st.session_state.user_data[f"CB_All_{food}"] = feedback
                    st.divider()
            else:
                st.info("Not enough data for content-based recommendations.")
        
        with tab6:
            st.write("**Method:** Similar users' preferences")
            if not historical_df.empty:
                top_5 = sorted(collaborative_scores_all.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_5 and top_5[0][1] > 0:
                    for i, (food, score) in enumerate(top_5, 1):
                        st.markdown(f"**{i}. {food}** (Score: {score:.3f})")
                        st.write("*Ingredients:* " + ", ".join(all_cuisine_foods.get(food, [])))
                        feedback = st.radio(
                            "Do you like this recommendation?",
                            ("👍 Thumbs Up", "👎 Thumbs Down"),
                            key=f"collab_all_{i}",
                            horizontal=True
                        )
                        st.session_state.user_data[f"Collab_All_{food}"] = feedback
                        st.divider()
                else:
                    st.info("Not enough similar users found.")
            else:
                st.info("No historical data available yet. Collaborative filtering requires multiple users.")
    else:
        st.info("Please rate at least one food with 4 or 5 stars to receive recommendations.")
    
    # --- SAVE DATA --- #
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
        
        # Get existing headers
        existing_headers = sheet.row_values(1)
        all_keys = list(st.session_state.user_data.keys())

        # Add any new keys to the sheet header
        for key in all_keys:
            if key not in existing_headers:
                existing_headers.append(key)

        # Update headers
        if not existing_headers:
            sheet.append_row(all_keys)
        else:
            sheet.update('A1', [existing_headers])
        
        # Prepare data row matching header order
        row_data = [str(st.session_state.user_data.get(h, "")) for h in existing_headers]
        sheet.append_row(row_data)
        
        st.success("✅ Your responses have been saved successfully!")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")
        
        # Fallback: save to CSV locally
        df_new = pd.DataFrame([st.session_state.user_data])
        try:
            df_existing = pd.read_csv(DATA_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except FileNotFoundError:
            df_combined = df_new
        df_combined.to_csv(DATA_FILE, index=False)
        st.warning("Data saved to local CSV as backup.")
    
    # Summary
    st.markdown("""
    ---
    ### Summary of Your Participation:
    - **Foods Rated:** {} foods from {} cuisine(s)
    - **Your Dietary Tolerance Level:** {} mg/day
    
    Thank you for contributing to PKU dietary research!
    
    Your responses will help develop better food recommendation systems for PKU patients.
    """.format(
        len(st.session_state.selected_foods),
        len(st.session_state.selected_cuisines),
        st.session_state.user_data.get("Dietary_tolerance_mg_per_day", "N/A")
    ))
    
    if st.button("Start New Response"):
        # Clear cache to reload fresh data
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()