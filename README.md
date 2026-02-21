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
**Pipeline only (Stage 0–7 + 5-fold), or if full install fails (e.g. Python 3.14 / Rust):**
`pip install -r requirements-pipeline.txt` then `python run_full_pipeline.py`

### System Architecture
+ Stage 0: Data preparation and cleaning from USDA database
+ Stage 2c: Meal clustering (ingredient-based)
+ Stage 1: Train-test split of user rating data (80/20)
+ Stage 2: Generate recommendations using hybrid algorithm
+ Stage 3: Calculate portions based on nutritional limits
+ Stage 4: Evaluate performance (precision, recall, F1, hit rate)
+ Stage 5–7: Combined evaluation, statistical tests, visualization
+ **5-fold cross-validation** (robustness): runs after Stage 1; reports mean ± std over 5 folds, then restores 80/20 and re-runs Stages 2–7.

**Run everything in order (Stage 0 → 2c → 1 → 5-fold, then 2–7 inside 5-fold):**
```bash
python run_full_pipeline.py
```
Or run stages manually; for 5-fold only (after 0 and 2c): `python run_5fold_cross_validation.py`.

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

