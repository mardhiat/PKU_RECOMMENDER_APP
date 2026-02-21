# Table 3: Fill-in Values and 5-Fold Cross-Validation Review

## 1. Numbers to put in Table 3

Use these values in **Table 3** (5-fold user-wise cross-validation, mean ± standard deviation).

### F1@10 (%) — from 5-fold CV (use these)

| Algorithm | F1@10 (%) |
|-----------|------------|
| Collaborative (Cuisine-Filtered) | **15.2 ± 1.3** |
| Hybrid (Cuisine-Filtered)         | **14.2 ± 1.0** |
| Content-Based (Cuisine-Filtered) | **10.1 ± 1.0** |

Source: `cross_validation_5fold_results.csv` (mean ± std over 5 folds).

### Liked & Safe (%) — from single 80/20 only (no 5-fold yet)

The current 5-fold script does **not** compute “Liked & Safe” per fold (it only runs Stage 2 + Stage 4 per fold, not Stage 3 + Stage 5). So we only have **single-split** Liked & Safe from Stage 5:

| Algorithm | Liked & Safe (%) (80/20 single split) |
|-----------|----------------------------------------|
| Collaborative (Cuisine-Filtered) | 3.7 |
| Hybrid (Cuisine-Filtered)         | 3.6 |
| Content-Based (Cuisine-Filtered) | 3.3 |

**Options for the paper:**

- **Option A (recommended):** In Table 3, fill **F1@10** with the 5-fold values above. For **Liked & Safe**, either:
  - use the single-split values (3.7, 3.6, 3.3) and add a footnote: *“Liked & Safe from primary 80/20 evaluation; F1@10 from 5-fold cross-validation.”*, or  
  - keep the column as “Liked & Safe (%, 80/20)” in the caption and use 3.7, 3.6, 3.3 without ± (single split).
- **Option B:** Extend the 5-fold pipeline to run Stage 3 and Stage 5 in each fold and report mean ± std for Liked & Safe as well (then both columns would be 5-fold).

---

## 2. Comprehensive review: what the 5-fold cross-validation did

### Purpose

- Complement the **single 80/20 split** with a **robustness check**.
- Ensure results are **stable across different train/test splits** and that **every rating is used as test exactly once**.

### Design

- **User-wise, stratified 5-fold:**
  - For each user, their ratings are split into 5 folds.
  - Splits are **stratified** by “liked” (rating ≥ 4) vs “not liked” (rating &lt; 3) so each fold keeps a similar proportion of liked/not-liked.
  - In each fold, **one fifth** of each user’s ratings is the **test set**; the other four fifths are **train**.
  - After 5 folds, **every rating has been in the test set exactly once** (and in train four times).

- **Per fold:**
  1. Train/test CSVs are written for that fold (80% train, 20% test per user).
  2. **Stage 2** (recommendations) is run on that train set; recommendations are produced for test users.
  3. **Stage 4** (preference evaluation) is run: Precision@10, Recall@10, F1@10, Hit Rate are computed for that fold’s test set.
  4. These four metrics are stored for the fold.

- **Aggregation:**
  - For each algorithm, the **mean** and **standard deviation** of each metric over the 5 folds are computed.
  - Output: `cross_validation_5fold_results.csv` with columns like  
    `Algorithm`, `Precision@10 (%)`, `Recall@10 (%)`, `F1@10 (%)`, `Hit Rate (%)`,  
    each cell as **mean ± std**.

### What is and is not computed in 5-fold

- **Computed in 5-fold (and reported in the CSV):**
  - **Precision@10, Recall@10, F1@10, Hit Rate** for all algorithms (preference-only metrics; no safety in this step).
- **Not computed in 5-fold (by design of the current script):**
  - **Liked & Safe %, Coverage %, Acceptance %** (Stage 5 metrics). These require Stage 3 (portion/safety) and Stage 5 per fold; the current pipeline only runs Stage 2 and Stage 4 per fold.

### Interpretation

- **F1@10 (5-fold):**  
  - Collaborative (Cuisine-Filtered): **15.2 ± 1.3%** — best on average; std 1.3% indicates stable performance across folds.  
  - Hybrid (Cuisine-Filtered): **14.2 ± 1.0%** — close second, slightly lower mean, similar stability.  
  - Content-Based (Cuisine-Filtered): **10.1 ± 1.0%** — clearly lower than collaborative and hybrid but still much better than baselines.

- **Stability:**  
  The standard deviations (about 1.0–1.3% for F1@10 for the top two) are small relative to the means, so the **ranking of algorithms and the benefit of cuisine filtering** are **consistent across folds**, not driven by one lucky split.

- **Relation to the paper:**
  - The 5-fold **F1@10** (and optionally Precision/Recall/Hit Rate) can be reported as the **robustness** result.
  - **Liked & Safe** in Table 3 can be filled from the **primary 80/20** run (Option A above) until the pipeline is extended to compute it in 5-fold (Option B).

### Summary sentence for the paper

*“We performed 5-fold user-wise cross-validation (stratified by liked/not-liked per user); each rating was used for testing exactly once. Recommendation models were retrained per fold; we report mean ± standard deviation of F1@10 (and optionally Precision@10, Recall@10, Hit Rate) across folds. Results confirm that cuisine-filtered collaborative and hybrid models perform best, with stable estimates across folds.”*

Use the **F1@10** values from the first table above to update Table 3; use the **Liked & Safe** values and footnote as in Option A unless you later add 5-fold safety metrics (Option B).
