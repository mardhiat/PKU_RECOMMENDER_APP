# Data

This repository no longer contains study participant records. Do not add names, emails, ages tied to a real person, or the original Google Sheet export.

## What is public

| Path | Contents |
| --- | --- |
| `data/cuisines/` | Ten cuisine tables, one shared column layout |
| `data/nutrition/` | Ingredient nutrition, a chat-derived ingredient table, and baby-food reference values |
| `data/processed/` | Meal nutrition, clusters, and ingredient matches built from those tables |
| `data/fixtures/synthetic_ratings.csv` | 12 fictional users (`demo-01` … `demo-12`) so the pipeline can run |
| `docs/results/` | Aggregate metrics from the original study. No names and no emails |
| `docs/figures/` | Figures for those aggregates |

The code is MIT. The food tables are reference data for this project. They are not a clinical database and they are not a license to redistribute participant records.

## Phenylalanine values

Some ingredient matches have no measured phenylalanine. When protein is present and phenylalanine is missing, preparation estimates 50 mg phenylalanine per gram of protein and should record `phe_source=imputed_50mg_per_g_protein`. Rows already in `food_database.csv` are marked `phe_source=legacy_unflagged` because that flag did not exist when they were built. A zero for both phenylalanine and protein means the match failed. It does not mean the food is free of phenylalanine.

`data/nutrition/chat_ingredients.csv` still contains unmatched ingredients stored as zeros. Treat those zeros as missing.

## Reproducing the published numbers

`docs/results/` is the study result. Running `python pipeline/run_pipeline.py` uses the synthetic fixture, so the new metrics will differ. That run checks that the code works. It does not recreate the manuscript table.

## If you cloned an older copy

The history on `main` was rewritten, so the participant export is not in these commits. A clone from before that rewrite still has the old files on disk. Re-clone, and rotate any API key that was ever in the sheet or in a secrets file.
