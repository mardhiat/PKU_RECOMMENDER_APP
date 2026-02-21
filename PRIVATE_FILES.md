# Files not pushed to public repo (in .gitignore)

These are ignored so they are **not** committed when you push publicly.

## Secrets and credentials
- `.streamlit/secrets.toml` – USDA API key, Gemini/API keys
- `.env`, `.env.*` – environment variables and secrets

## Caches
- `usda_cache.json` – USDA API response cache

## Research / participant data
- `data_user_food_ratings.csv` – user names and ratings
- `data_user_nutritional_limits.csv` – user profiles/limits
- `data_train_ratings.csv`, `data_test_ratings.csv`, `data_test_users.csv` – derived splits (contain user identifiers)

---

## If these were already committed

To stop tracking them **without deleting** the files on your machine:

```bash
git rm --cached .streamlit/secrets.toml 2>nul
git rm --cached usda_cache.json 2>nul
git rm --cached data_user_food_ratings.csv data_user_nutritional_limits.csv 2>nul
git rm --cached data_train_ratings.csv data_test_ratings.csv data_test_users.csv 2>nul
git commit -m "Stop tracking sensitive and research data"
```

Then push. The files stay on your disk but won’t be in the repo. If they were pushed before, consider rotating any API keys that were in them.
