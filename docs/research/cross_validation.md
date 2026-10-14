# 5-fold cross-validation

User-wise, stratified 5-fold check on the original study ratings. Each person's ratings were split so that every rating was in the test set once. Models were retrained each fold. A liked food is a rating of 4 or 5.

The single 80/20 split and this 5-fold check answer different questions. Both are in `docs/results/`.

| Algorithm | F1@10 (%) | Liked and safe (%) |
| --- | --- | --- |
| Collaborative, cuisine-filtered | 14.8 ± 1.5 | 3.7 ± 0.8 |
| Hybrid, cuisine-filtered | 13.7 ± 1.5 | 3.7 ± 0.8 |
| Content-based, cuisine-filtered | 9.7 ± 0.7 | 2.8 ± 0.3 |

Source: `docs/results/cross_validation.csv`. On the single split, cuisine-filtered collaborative filtering leads on F1@10 (13.0%) and on liked-and-safe (3.7%). Hybrid is close behind on F1@10 (12.8%).

Cuisine filtering is not uniformly p < 0.001. Content-based filtering improves (p < 0.001). Hybrid improves (p = 0.0018). Collaborative filtering's cuisine comparison is p = 0.058. Three of eight paired comparisons in the published test file are significant at 0.05, and those tests are not corrected for multiple comparisons.
