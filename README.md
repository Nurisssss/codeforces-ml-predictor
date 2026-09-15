# codeforces-ml-predictor

## Model Performance Log
* **Date:** 2026-09-14
* **Dataset Size:** 1959 problems
* **Model:** Random Forest (Baseline TF-IDF)
* **Mean Absolute Error:** 419.01 points

* **Date:** 2026-09-15
* **Dataset Size:** 1959 problems
* **Model:** Random Forest (TF-IDF + Word Count Feature)
* **Mean Absolute Error:** 419.85 points
* **Notes:** Added word count as a numerical feature, but the MAE slightly increased. This suggests raw text length does not reliably dictate problem difficulty.