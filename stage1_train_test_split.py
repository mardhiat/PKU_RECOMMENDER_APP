import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

 
print("STAGE 1: CREATE TRAIN/TEST SPLITS (FIXED)")
 

 # CONFIGURATION
 
MIN_RATINGS_PER_USER = 10  # Minimum ratings to include user
MIN_LIKED_FOODS = 3        # Minimum liked foods (rating >= 3)
TEST_SIZE = 0.2            # 20% test, 80% train
RANDOM_STATE = 42

 # STEP 1.1: LOAD DATA FROM STAGE 0
 
print("\nSTEP 1.1: LOADING DATA FROM STAGE 0\n")

try:
    ratings_df = pd.read_csv('data_user_food_ratings.csv')
    user_limits_df = pd.read_csv('data_user_nutritional_limits.csv')
    food_db = pd.read_csv('data_food_database.csv')
    
    print("Loaded data files:")
    print(f"  - User ratings: {len(ratings_df)} ratings")
    print(f"  - User limits: {len(user_limits_df)} users")
    print(f"  - Food database: {len(food_db)} foods")
except FileNotFoundError as e:
    print(f"ERROR: Missing file - {e}")
    print("Please run stage0_data_preparation.py first")
    exit(1)

 # STEP 1.2: FILTER FOODS WITH NUTRITIONAL DATA (CASE-INSENSITIVE)
 
print("\nSTEP 1.2: FILTERING FOODS WITH NUTRITIONAL DATA\n")

# CRITICAL FIX: Case-insensitive matching
food_db['food_name_lower'] = food_db['food_name'].str.lower().str.strip()
ratings_df['food_lower'] = ratings_df['food'].str.lower().str.strip()

# Count foods with complete nutrition data
foods_with_nutrition = food_db[
    (food_db['phe_mg_per_100g'] > 0) | 
    (food_db['protein_g_per_100g'] > 0) |
    (food_db['data_quality'].isin(['complete', 'partial']))
]

print(f"Foods with nutritional data: {len(foods_with_nutrition)}")

# Filter ratings to only include foods with nutrition data
valid_foods_lower = set(foods_with_nutrition['food_name_lower'].values)

print(f"\nBefore filtering: {len(ratings_df)} ratings")

filtered_ratings = ratings_df[ratings_df['food_lower'].isin(valid_foods_lower)].copy()

# Drop the temporary lowercase column
filtered_ratings = filtered_ratings.drop(columns=['food_lower'])

print(f"After filtering: {len(filtered_ratings)} ratings")
print(f"Removed: {len(ratings_df) - len(filtered_ratings)} ratings without nutrition data")

if len(filtered_ratings) == 0:
    print("\nERROR: No ratings remain after filtering!")
    print("\nDEBUGGING INFO:")
    print(f"Sample food names from ratings (first 5):")
    for food in ratings_df['food'].head():
        print(f"  - '{food}' -> '{food.lower().strip()}'")
    print(f"\nSample food names from database (first 5):")
    for food in food_db['food_name'].head():
        print(f"  - '{food}' -> '{food.lower().strip()}'")
    exit(1)

 # STEP 1.3: FILTER USERS WITH SUFFICIENT DATA
 
print(f"\nSTEP 1.3: FILTERING USERS WITH SUFFICIENT DATA\n")

print(f"Minimum requirements:")
print(f"  - At least {MIN_RATINGS_PER_USER} ratings (with nutrition data)")
print(f"  - At least {MIN_LIKED_FOODS} liked foods (rating >= 3)")

# Calculate per-user statistics
user_stats = filtered_ratings.groupby('user_name').agg({
    'rating': ['count', lambda x: (x >= 4).sum()]  # FIXED: Changed from 4 to 3
}).reset_index()

user_stats.columns = ['user_name', 'total_ratings', 'liked_foods']

# Filter users meeting minimum requirements
eligible_users = user_stats[
    (user_stats['total_ratings'] >= MIN_RATINGS_PER_USER) &
    (user_stats['liked_foods'] >= MIN_LIKED_FOODS)
]['user_name'].values

print(f"\nUser statistics:")
print(f"  Total users: {len(user_stats)}")
print(f"  Mean ratings per user: {user_stats['total_ratings'].mean():.1f}")
print(f"  Mean liked foods per user: {user_stats['liked_foods'].mean():.1f}")

print(f"\nEligible users: {len(eligible_users)}")
print(f"  Removed users: {len(user_stats) - len(eligible_users)}")

if len(eligible_users) == 0:
    print("\nERROR: No users meet the minimum requirements!")
    print(f"Try lowering MIN_RATINGS_PER_USER or MIN_LIKED_FOODS")
    
    # Show distribution
    print(f"\nRatings distribution:")
    print(user_stats[['total_ratings', 'liked_foods']].describe())
    exit(1)

# Filter ratings to only eligible users
filtered_ratings = filtered_ratings[
    filtered_ratings['user_name'].isin(eligible_users)
].copy()

# REMOVE DUPLICATES
print(f"\nChecking for duplicate ratings...")
before_dedup = len(filtered_ratings)
filtered_ratings = filtered_ratings.drop_duplicates(subset=['user_name', 'food'], keep='first')
after_dedup = len(filtered_ratings)

if before_dedup > after_dedup:
    print(f"  Removed {before_dedup - after_dedup} duplicate user-food pairs")
else:
    print(f"  No duplicates found")

print(f"\nFinal dataset: {len(filtered_ratings)} ratings from {len(eligible_users)} users")

 # STEP 1.4: CREATE TRAIN/TEST SPLIT
 
print(f"\nSTEP 1.4: CREATING TRAIN/TEST SPLIT\n")

train_list = []
test_list = []

for user_name in eligible_users:
    user_ratings = filtered_ratings[filtered_ratings['user_name'] == user_name]
    
    if len(user_ratings) < 5:
        # Too few ratings - put all in training
        train_list.append(user_ratings)
    else:
        # Stratified split (liked vs not-liked)
        liked = user_ratings[user_ratings['rating'] >= 4]
        not_liked = user_ratings[user_ratings['rating'] < 3]
        
        # Split each group
        train_parts = []
        test_parts = []
        
        if len(liked) >= 2:
            liked_train, liked_test = train_test_split(
                liked, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            train_parts.append(liked_train)
            test_parts.append(liked_test)
        elif len(liked) == 1:
            train_parts.append(liked)
        
        if len(not_liked) >= 2:
            not_liked_train, not_liked_test = train_test_split(
                not_liked, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            train_parts.append(not_liked_train)
            test_parts.append(not_liked_test)
        elif len(not_liked) == 1:
            train_parts.append(not_liked)
        
        if train_parts:
            train_list.append(pd.concat(train_parts, ignore_index=True))
        if test_parts:
            test_list.append(pd.concat(test_parts, ignore_index=True))

train_df = pd.concat(train_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)

print(f"Train set: {len(train_df)} ratings from {train_df['user_name'].nunique()} users")
print(f"Test set: {len(test_df)} ratings from {test_df['user_name'].nunique()} users")
print(f"Split ratio: {len(test_df)/(len(train_df)+len(test_df))*100:.1f}% test")

 # STEP 1.5: SAVE OUTPUTS
 
print(f"\nSTEP 1.5: SAVING OUTPUTS\n")

# FIXED: Use data_ prefix to match Stage 2-5
train_df.to_csv('data_train_ratings.csv', index=False)
test_df.to_csv('data_test_ratings.csv', index=False)

eligible_users_df = pd.DataFrame({'user_name': eligible_users})
eligible_users_df.to_csv('data_test_users.csv', index=False)

print(f"OK Saved: data_train_ratings.csv")
print(f"OK Saved: data_test_ratings.csv")
print(f"OK Saved: data_test_users.csv")

 # SUMMARY
 
  
print("STAGE 1 COMPLETE - SUMMARY")
print(f"{'='*70}\n")

print(f"DATASET STATISTICS:")
print(f"  Total ratings (filtered): {len(filtered_ratings)}")
print(f"  Eligible users: {len(eligible_users)}")
print(f"  Foods with ratings: {filtered_ratings['food'].nunique()}")
print(f"  Foods with nutrition data: {len(foods_with_nutrition)}")

print(f"\nTRAIN/TEST SPLIT:")
print(f"  Train: {len(train_df)} ratings ({len(train_df)/(len(train_df)+len(test_df))*100:.1f}%)")
print(f"  Test: {len(test_df)} ratings ({len(test_df)/(len(train_df)+len(test_df))*100:.1f}%)")

print(f"\nRATING DISTRIBUTION (TRAIN):")
print(train_df['rating'].value_counts().sort_index())

print(f"\nRATING DISTRIBUTION (TEST):")
print(test_df['rating'].value_counts().sort_index())

  
print("NEXT: Run stage2_generate_recommendations.py")
  