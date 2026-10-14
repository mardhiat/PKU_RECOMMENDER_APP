# PKU meal planner and recommender evaluation

A Streamlit meal planner for phenylketonuria (PKU) diet planning, plus an offline evaluation of cuisine-aware recommenders. The work was done at Kean University's School of Integrative Science and Technology. The manuscript is in preparation for BMC Medical Informatics and Decision Making.

The app supports planning. It does not replace a metabolic clinic.

## Run the app

Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The planner loads `data/nutrition/Nutritional_Data.csv` and the cuisine tables in `data/cuisines/`. A profile stores nothing on a server. The meal plan can be downloaded as CSV.

## Run the evaluation

```bash
pip install -r requirements-pipeline.txt
python pipeline/run_pipeline.py
```

That command uses `data/fixtures/synthetic_ratings.csv` (fictional `demo-` users). It does not call the USDA API unless you pass `--refresh-nutrition` to `pipeline/prepare_data.py` and set `USDA_API_KEY` (see `.env.example`).

Optional parameter search, after recommendations exist:

```bash
python pipeline/tune_parameters.py
```

### Pipeline order

1. `prepare_data.py` reads the fixture and the cuisine tables.
2. `cluster_meals.py` clusters the meal catalog. Clusters describe foods, so this runs before any user split.
3. `split_ratings.py` makes an 80/20 user-wise split. A liked food is a rating of 4 or 5. A rating of 3 is left out of that split. A user needs at least 10 ratings and 3 liked foods.
4. `cross_validate.py` runs a 5-fold check, restores the 80/20 split, and then recommends, scores portions, evaluates, tests, and draws figures.

## What the study found

These numbers are the published aggregates in `docs/results/`. They are not the output of the synthetic fixture.

On the 80/20 split (102 users), cuisine-filtered collaborative filtering leads:

| Algorithm | F1@10 (%) | Precision@10 (%) | Recall@10 (%) | Hit rate (%) |
| --- | ---: | ---: | ---: | ---: |
| Collaborative, cuisine-filtered | 13.0 | 9.5 | 27.3 | 50.0 |
| Hybrid, cuisine-filtered | 12.8 | 8.9 | 30.3 | 52.0 |
| Content-based, cuisine-filtered | 11.0 | 7.4 | 27.4 | 50.0 |
| Popularity, cuisine-filtered | 10.5 | 6.9 | 28.1 | 49.0 |
| Random | 0.5 | 0.5 | 0.7 | 3.9 |

Liked-and-safe (liked, and a portion that stays inside the phenylalanine limit) is 3.7% for cuisine-filtered collaborative filtering. Scores are modest because each user has rated only a small set of foods.

Every algorithm in that evaluation recommends only from foods someone has already rated. It is not a cold-start score over the full catalog. There are 10 cuisine tables, about 640 composed meals, and about 800 ingredient rows in the nutrition table.

Five-fold means and the per-comparison p-values are in [docs/research/cross_validation.md](docs/research/cross_validation.md). Cuisine filtering helps content-based filtering (p < 0.001) and hybrid (p = 0.0018). The collaborative comparison is p = 0.058.

## Repository map

| Path | Role |
| --- | --- |
| `app.py` | Meal planner |
| `pipeline/` | Evaluation scripts |
| `pku/` | Shared age, phenylalanine, and path helpers |
| `data/` | Food tables and the synthetic rating fixture |
| `docs/results/` | Study metrics |
| `docs/figures/` | Study figures |
| `DATA.md` | What is safe to publish |

## Contact

Mardhiat Ajetunmobi  
mardhiata@gmail.com  
[LinkedIn](https://www.linkedin.com/in/mardhiat-ajetunmobi)
