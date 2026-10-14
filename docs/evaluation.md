# Evaluation figures

Figures are written to `docs/figures/` when `pipeline/visualize.py` runs.

| File | What it shows |
| --- | --- |
| `preference_metrics.png` | Precision@10, recall@10, and F1@10 |
| `liked_and_safe.png` | Share of recommendations that are both liked and within the PHE portion limit |
| `cuisine_filtering.png` | Cuisine-filtered vs all-cuisine variants, with the per-comparison p-value |
| `three_perspectives.png` | Liked-and-safe, coverage, and acceptance side by side |
| `summary_table.png` | The same metrics in one table |

Stars on the cuisine figure mark one comparison at a time: `***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `ns` otherwise. They are not a single p-value for every algorithm. In the published run, content-based filtering shows a significant cuisine effect. Collaborative filtering does not (p = 0.058). See `docs/results/summary.txt`.

Green bars are cuisine-filtered. Gray bars use the full catalog. Highlighted cells in the summary table are the best value in that column.

These scores are computed only on foods that already have ratings. They are not a cold-start result over every food in the database.
