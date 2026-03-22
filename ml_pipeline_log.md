# ML Pipeline Log for NYC Airbnb Gentrification

## Setup - Data Preparation
- Initial rows: 2116
- Rows after dropping nulls: 1508

## Model 1 — Decision Tree Classifier
**Metrics:**
- Accuracy: 0.9294
- F1 Score: 0.8881

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.96      0.93      0.95       315
           1       0.86      0.92      0.89       138

    accuracy                           0.93       453
   macro avg       0.91      0.93      0.92       453
weighted avg       0.93      0.93      0.93       453

```
**Top 3 Features (DT):**
- pct_white: 0.6623
- median_income: 0.2657
- poverty_pct: 0.0612

## Model 2 — Random Forest Classifier
**Metrics:**
- Accuracy: 0.9735
- F1 Score: 0.9562
- ROC-AUC: 0.9967

**Confusion Matrix:**
```
[[310   5]
 [  7 131]]
```
**Performance Note:** Random Forest outperformed Decision Tree by 4.42% accuracy points.

## Model 3 — K-Means Clustering
- Elbow plot generated. Recommended k is typically the point of inflection (elbow). Fit performed with k=4 as requested.
**Cluster Profile Table (Mean Values):**

|                                          |   listings_per_1000_residents |   median_income |   pct_white |   poverty_pct |   evictions_per_1000_residents |   gentrification_proxy |
|:-----------------------------------------|------------------------------:|----------------:|------------:|--------------:|-------------------------------:|-----------------------:|
| Cluster 0: Affluent Low-Tourism          |                       4.06233 |         86404.6 |    68.092   |       9.36381 |                        1.59927 |               2.8787   |
| Cluster 1: Stable Low-Pressure           |                       3.38944 |         56702.3 |    21.1701  |      16.4385  |                        3.11403 |              -0.236586 |
| Cluster 2: Displacement Risk             |                       3.89833 |         31618.7 |     9.74663 |      34.8751  |                        7.68781 |              -2.88191  |
| Cluster 3: Tourism Hotspot / Gentrifying |                      36.1246  |        101647   |    62.132   |      13.132   |                        3.26661 |               2.93064  |
- Choropleth map skipped: 'geometry' column not found in dataframe.

## Summary

### Model Comparison
| Metric | Decision Tree | Random Forest |
| :--- | :--- | :--- |
| Accuracy | 0.9294 | 0.9735 |
| F1 Score | 0.8881 | 0.9562 |
| ROC-AUC | N/A | 0.9967 |

### Top 3 Features
- **Decision Tree:** pct_white, median_income, poverty_pct
- **Random Forest:** median_income, pct_white, poverty_pct

### K-Means Interpretation
The K-Means clustering revealed natural divisions in NYC tracts, distinguishing areas with high tourism pressure and gentrification from those facing high displacement risks (evictions) and more stable, affluent, or low-pressure neighborhoods. The clusters correlate strongly with racial and economic demographics.

### Output Files
- `decision_tree_plot.png`: Tree visualization.
- `decision_tree_importances.png`: DT feature rankings.
- `random_forest_roc.png`: RF ROC curve.
- `random_forest_importances.png`: RF feature rankings.
- `importances_comparison.png`: Side-by-side DT/RF importance comparison.
- `kmeans_elbow.png`: Optimal k selection plot.
- `kmeans_cluster_heatmap.png`: Standardized cluster profiles.
- `kmeans_scatter.png`: Airbnb vs Proxy scatter by cluster.
- `ml_pipeline_log.md`: Full event and results log.

