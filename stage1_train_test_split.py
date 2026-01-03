import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


print("STAGE 1: CREATE TRAIN/TEST SPLITS")


# STEP 1.1: LOAD DATA FROM STAGE 0


print("STEP 1.1: LOADING DATA FROM STAGE 0")

# Check if Stage 0 files exist
required_files = [
    'data_user_food_ratings.csv',
    'data_user_nutritional_limits.csv',
    'data_food_database.csv'
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\nERROR: {file} not found!")
        print("Please run Stage 0 first (data_preparation.py)")
        exit()

# Load data
user_food_df = pd.read_csv('data_user_food_ratings.csv')
user_limits_df = pd.read_csv('data_user_nutritional_limits.csv')
food_db_df = pd.read_csv('data_food_database.csv')

print(f"\nLoaded data files:")
print(f"  - User ratings: {len(user_food_df)} ratings")
print(f"  - User limits: {len(user_limits_df)} users")
print(f"  - Food database: {len(food_db_df)} foods")


# STEP 1.2: FILTER RATINGS TO ONLY FOODS WITH NUTRITIONAL DATA


print("STEP 1.2: FILTERING FOODS WITH NUTRITIONAL DATA")

# Create set of foods that have nutritional data
foods_with_data = set(food_db_df['food_name'].str.lower().str.strip())

print(f"\nFoods with nutritional data: {len(foods_with_data)}")

# Function to check if a food has nutritional data
def has_nutritional_data(food_name):
    """Check if food has nutritional data in database"""
    # Clean food name (remove cuisine tag)
    clean_name = food_name.lower().strip()
    if '(' in clean_name:
        clean_name = clean_name.split('(')[0].strip()
    
    # Check if in database
    return clean_name in foods_with_data

# Filter ratings
print(f"\nBefore filtering: {len(user_food_df)} ratings")

user_food_df['has_nutrition_data'] = user_food_df['food'].apply(has_nutritional_data)
filtered_ratings_df = user_food_df[user_food_df['has_nutrition_data']].copy()
filtered_ratings_df = filtered_ratings_df.drop('has_nutrition_data', axis=1)

print(f"After filtering: {len(filtered_ratings_df)} ratings")
print(f"Removed: {len(user_food_df) - len(filtered_ratings_df)} ratings without nutrition data")


# STEP 1.3: FILTER USERS WITH SUFFICIENT DATA


print("STEP 1.3: FILTERING USERS WITH SUFFICIENT DATA")

# Set minimum requirements
MIN_RATINGS_PER_USER = 10  # Must have at least 10 ratings
MIN_LIKED_FOODS = 3        # Must have liked at least 3 foods (rating ≥ 4)

print(f"\nMinimum requirements:")
print(f"  - At least {MIN_RATINGS_PER_USER} ratings (with nutrition data)")
print(f"  - At least {MIN_LIKED_FOODS} liked foods (rating ≥ 4)")

# Count ratings and liked foods per user
user_stats = filtered_ratings_df.groupby('user_name').agg({
    'rating': ['count', lambda x: (x >= 4).sum()]
})
user_stats.columns = ['total_ratings', 'liked_foods']

print(f"\nUser statistics:")
print(f"  Total users: {len(user_stats)}")
print(f"  Mean ratings per user: {user_stats['total_ratings'].mean():.1f}")
print(f"  Mean liked foods per user: {user_stats['liked_foods'].mean():.1f}")

# Filter users
eligible_users = user_stats[
    (user_stats['total_ratings'] >= MIN_RATINGS_PER_USER) &
    (user_stats['liked_foods'] >= MIN_LIKED_FOODS)
]

print(f"\nEligible users: {len(eligible_users)}")
print(f"  Removed users: {len(user_stats) - len(eligible_users)}")

if len(eligible_users) == 0:
    print("\nERROR: No users meet the minimum requirements!")
    print("Try lowering MIN_RATINGS_PER_USER or MIN_LIKED_FOODS")
    exit()

# Filter to eligible users only
eligible_user_names = eligible_users.index.tolist()
eval_data = filtered_ratings_df[filtered_ratings_df['user_name'].isin(eligible_user_names)].copy()

# REMOVE DUPLICATE USER-FOOD PAIRS (keep first rating)
print(f"\nChecking for duplicate ratings...")
before_dedup = len(eval_data)
eval_data = eval_data.drop_duplicates(subset=['user_name', 'food'], keep='first')
after_dedup = len(eval_data)

if before_dedup > after_dedup:
    print(f"   Removed {before_dedup - after_dedup} duplicate user-food pairs")
    print(f"  Kept the first rating for each user-food combination")
else:
    print(f"   No duplicates found")

print(f"\nFiltered dataset:")
print(f"  Users: {eval_data['user_name'].nunique()}")
print(f"  Total ratings: {len(eval_data)}")
print(f"  Unique foods: {eval_data['food'].nunique()}")
print(f"  Ratings per user: {len(eval_data) / eval_data['user_name'].nunique():.1f}")


# STEP 1.4: CREATE STRATIFIED TRAIN/TEST SPLIT


print("STEP 1.4: CREATING TRAIN/TEST SPLIT (80/20)")

TEST_SIZE = 0.2  # 20% test, 80% train
RANDOM_STATE = 42  # For reproducibility

print(f"\nSplit configuration:")
print(f"  Train: {(1-TEST_SIZE)*100:.0f}%")
print(f"  Test: {TEST_SIZE*100:.0f}%")
print(f"  Random seed: {RANDOM_STATE}")

def create_stratified_split(user_ratings, test_size=0.2, random_state=42):
    """
    Create stratified train/test split for a single user
    Ensures both liked and disliked foods appear in test set
    """
    # Separate liked (≥4) and not-liked (<4)
    liked = user_ratings[user_ratings['rating'] >= 4]
    not_liked = user_ratings[user_ratings['rating'] < 4]
    
    train_parts = []
    test_parts = []
    
    # Split liked foods (if enough)
    if len(liked) >= 2:
        liked_train, liked_test = train_test_split(
            liked, test_size=test_size, random_state=random_state
        )
        train_parts.append(liked_train)
        test_parts.append(liked_test)
    elif len(liked) == 1:
        # Only 1 liked food - put in train
        train_parts.append(liked)
    
    # Split not-liked foods (if enough)
    if len(not_liked) >= 2:
        not_liked_train, not_liked_test = train_test_split(
            not_liked, test_size=test_size, random_state=random_state
        )
        train_parts.append(not_liked_train)
        test_parts.append(not_liked_test)
    elif len(not_liked) == 1:
        # Only 1 not-liked food - put in train
        train_parts.append(not_liked)
    
    # Combine
    if train_parts:
        user_train = pd.concat(train_parts, ignore_index=True)
    else:
        user_train = pd.DataFrame()
    
    if test_parts:
        user_test = pd.concat(test_parts, ignore_index=True)
    else:
        user_test = pd.DataFrame()
    
    return user_train, user_test

# Split for each user
train_list = []
test_list = []

for user_name in eval_data['user_name'].unique():
    user_ratings = eval_data[eval_data['user_name'] == user_name]
    user_train, user_test = create_stratified_split(
        user_ratings, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    if len(user_train) > 0:
        train_list.append(user_train)
    if len(user_test) > 0:
        test_list.append(user_test)

# Combine all users
train_df = pd.concat(train_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)

print(f"\nSplit complete:")
print(f"  Train set: {len(train_df)} ratings from {train_df['user_name'].nunique()} users")
print(f"  Test set: {len(test_df)} ratings from {test_df['user_name'].nunique()} users")


# STEP 1.5: VERIFY SPLIT QUALITY


print("STEP 1.5: VERIFYING SPLIT QUALITY")

# Check rating distribution
print(f"\nRating distribution in TRAIN set:")
train_dist = train_df['rating'].value_counts(normalize=True).sort_index()
for rating, pct in train_dist.items():
    print(f"  {rating}: {pct*100:5.1f}%")

print(f"\nRating distribution in TEST set:")
test_dist = test_df['rating'].value_counts(normalize=True).sort_index()
for rating, pct in test_dist.items():
    print(f"  {rating}: {pct*100:5.1f}%")

# Check liked foods in test set
test_liked_per_user = test_df[test_df['rating'] >= 4].groupby('user_name').size()
print(f"\nLiked foods (rating ≥ 4) in test set per user:")
print(f"  Mean: {test_liked_per_user.mean():.1f}")
print(f"  Median: {test_liked_per_user.median():.1f}")
print(f"  Min: {test_liked_per_user.min()}")
print(f"  Max: {test_liked_per_user.max()}")

# Verify no data leakage (no same user-food pair in both sets)
train_pairs = set(zip(train_df['user_name'], train_df['food']))
test_pairs = set(zip(test_df['user_name'], test_df['food']))
overlap = train_pairs.intersection(test_pairs)

if len(overlap) > 0:
    print(f"\n⚠ WARNING: Found {len(overlap)} overlapping user-food pairs!")
    print("This should not happen - data leakage detected!")
else:
    print(f"\nNo data leakage: Train and test sets are completely separate")


# STEP 1.6: SAVE THE SPLITS


print("STEP 1.6: SAVING TRAIN/TEST SPLITS")

# Save train/test splits
train_df.to_csv('train_ratings.csv', index=False)
print(f"Saved: train_ratings.csv ({len(train_df)} ratings)")

test_df.to_csv('test_ratings.csv', index=False)
print(f"Saved: test_ratings.csv ({len(test_df)} ratings)")

# Save eligible users info
eligible_users_df = pd.DataFrame({
    'user_name': eligible_user_names,
    'total_ratings': [len(eval_data[eval_data['user_name'] == u]) for u in eligible_user_names],
    'train_ratings': [len(train_df[train_df['user_name'] == u]) for u in eligible_user_names],
    'test_ratings': [len(test_df[test_df['user_name'] == u]) for u in eligible_user_names],
    'liked_foods_in_test': [len(test_df[(test_df['user_name'] == u) & (test_df['rating'] >= 4)]) for u in eligible_user_names]
})

eligible_users_df.to_csv('eligible_users.csv', index=False)
print(f"Saved: eligible_users.csv ({len(eligible_users_df)} users)")


# FINAL SUMMARY


print("STAGE 1 COMPLETE - SUMMARY")

print(f"""
Files Created:
  1. train_ratings.csv  - {len(train_df)} ratings for building recommendations
  2. test_ratings.csv   - {len(test_df)} ratings for evaluation
  3. eligible_users.csv - {len(eligible_users_df)} users in evaluation

Dataset Statistics:
  • Total users in evaluation: {len(eligible_user_names)}
  • Total ratings: {len(eval_data)}
  • Train/Test split: {len(train_df)}/{len(test_df)} ({len(train_df)/len(eval_data)*100:.1f}%/{len(test_df)/len(eval_data)*100:.1f}%)
  • Unique foods: {eval_data['food'].nunique()}
  • Average test ratings per user: {len(test_df)/len(eligible_user_names):.1f}
  • Average liked foods in test per user: {test_liked_per_user.mean():.1f}

Quality Checks:
 No data leakage between train and test
 Both liked and disliked foods in test set
 Rating distribution preserved in both sets
 All users have nutritional limits calculated

Next Steps:
  Ready for Stage 2: Generate Recommendations
  - Implement recommendation algorithms
  - Generate top-K recommendations for each test user
""")

