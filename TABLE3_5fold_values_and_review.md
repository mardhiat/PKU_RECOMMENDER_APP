# Table 3: Fill-in Values and 5-Fold Cross-Validation Review

## 1. Numbers to put in Table 3

Use these values in **Table 3** (5-fold user-wise cross-validation, mean ± standard deviation).

### Table 3 — both columns from 5-fold CV

| Algorithm | F1@10 (%) | Liked & Safe (%) |
|-----------|------------|-------------------|
| Collaborative (Cuisine-Filtered) | **14.8 ± 1.5** | **3.7 ± 0.8** |
| Hybrid (Cuisine-Filtered)         | **13.7 ± 1.5** | **3.7 ± 0.8** |
| Content-Based (Cuisine-Filtered) | **9.7 ± 0.7** | **2.8 ± 0.3** |

Source: `cross_validation_5fold_results.csv` and `cross_validation_5fold_table3.csv` (mean ± std over 5 folds). Both F1@10 and Liked & Safe are computed per fold (Stages 2–5 per fold) and aggregated.

---

## 2. Comprehensive review: what the 5-fold cross-validation did

### Purpose

- Complement the **single 80/20 split** with a **robustness check**.
- Ensure results are **stable across different train/test splits** and that **every rating is used as test exactly once**.

### Design

- **User-wise, stratified 5-fold:**
  - For each user, their ratings are split into 5 folds.
  - Splits are **stratified** by "liked" (rating ≥ 4) vs "not liked" (rating &lt; 3) so each fold keeps a similar proportion of liked/not-liked.
  - In each fold, **one fifth** of each user's ratings is the **test set**; the other four fifths are **train**.
  - After 5 folds, **every rating has been in the test set exactly once** (and in train four times).

- **Per fold:**
  1. Train/test CSVs are written for that fold (80% train, 20% test per user).
  2. **Stage 2** (recommendations) is run on that train set; recommendations are produced for test users.
  3. **Stage 4** (preference evaluation) is run: Precision@10, Recall@10, F1@10, Hit Rate are computed for that fold's test set.
  4. **Stage 3** (portions/safety) and **Stage 5** (combined evaluation) are run so **Liked & Safe %** is computed per fold.
  5. All metrics (preference + Liked & Safe) are stored for the fold.

- **Aggregation:**
  - For each algorithm, the **mean** and **standard deviation** of each metric over the 5 folds are computed.
  - Output: `cross_validation_5fold_results.csv` with columns  
    `Algorithm`, `Precision@10 (%)`, `Recall@10 (%)`, `F1@10 (%)`, `Hit Rate (%)`, `Liked & Safe (%)`,  
    each cell as **mean ± std**. Table 3 subset: `cross_validation_5fold_table3.csv`.

### What is computed in 5-fold

- **Precision@10, Recall@10, F1@10, Hit Rate** (Stages 2 + 4 per fold).
- **Liked & Safe %** (Stages 2 + 3 + 5 per fold); mean ± std reported for Table 3.

### Interpretation

- **F1@10 (5-fold):**  
  - Collaborative (Cuisine-Filtered): **14.8 ± 1.5%** — best on average; std 1.5% indicates stable performance across folds.  
  - Hybrid (Cuisine-Filtered): **13.7 ± 1.5%** — close second, slightly lower mean, similar stability.  
  - Content-Based (Cuisine-Filtered): **9.7 ± 0.7%** — clearly lower than collaborative and hybrid but still much better than baselines.

- **Liked & Safe (5-fold):**  
  - Collaborative and Hybrid: **3.7 ± 0.8%** each; Content-Based: **2.8 ± 0.3%**. Safety-aware metric is stable across folds.

- **Stability:**  
  The standard deviations (about 1.5% for F1@10 for the top two) are small relative to the means, so the **ranking of algorithms and the benefit of cuisine filtering** are **consistent across folds**, not driven by one lucky split.

- **Relation to the paper:**
  - Table 3 reports **F1@10** and **Liked & Safe %** from 5-fold cross-validation (mean ± std). Both columns use the same folds.

### Summary sentence for the paper

*"We performed 5-fold user-wise cross-validation (stratified by liked/not-liked per user); each rating was used for testing exactly once. Recommendation models were retrained per fold; we report mean ± standard deviation of F1@10 and Liked & Safe % across folds. Results confirm that cuisine-filtered collaborative and hybrid models perform best, with stable estimates across folds."*

Use the values from the **Table 3** section above for both columns.
