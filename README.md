# PKU Dietary Recommender System
AI-powered meal recommendation system for Phenylketonuria (PKU) patients using hybrid collaborative and content-based filtering.

### Overview
Web application that provides personalized dietary recommendations for PKU patients based on nutritional constraints, user preferences, and collaborative filtering from community ratings. Built as part of research at Kean University's School of Integrative Science and Technology.

### Tech Stack
Language: Python  
Framework: Streamlit  
ML Libraries: Scikit-learn, Pandas, NumPy  
Data: USDA nutritional database (900+ foods across 12 cuisines)  

### Features
+ Personalized nutrition calculations based on age, weight, and diet type
+ Hybrid recommendation algorithm combining content-based and collaborative filtering
+ User rating system for adaptive learning
+ Meal planning with CSV export
+ Nutrition visualization and tracking

### Setup
```
pip install -r requirements.txt
streamlit run app.py
```

### System Architecture
+ Stage 0: Data preparation and cleaning from USDA database
+ Stage 1: Train-test split of user rating data
+ Stage 2: Generate recommendations using hybrid algorithm
+ Stage 3: Calculate portions based on nutritional limits
+ Stage 4: Evaluate performance using precision, recall, and NDCG metrics

### Algorithms
Five recommendation approaches implemented:

1. Hybrid (content-based + collaborative filtering)
2. Content-based (nutrient similarity)
3. Collaborative filtering (user preferences)
4. Popularity-based (most-rated foods)
5. Random (baseline)

Hybrid approach demonstrates best overall performance across evaluation metrics.

### Research Status
Manuscript in preparation for submission to BMC Medical Informatics and Decision Making.

### Contact  
Mardhiat Ajetunmobi  
mardhiata@gmail.com  
[LinkedIn](www.linkedin.com/in/mardhiat-ajetunmobi)

