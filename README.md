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

* **Date:** 2026-09-16
* **Dataset Size:** 1959 problems
* **Model:** Random Forest (Word2Vec Embeddings)
* **Mean Absolute Error:** 550.05 points
* **Notes:** Tested custom Word2Vec embeddings trained on the 1,959 problem corpus. The error increased significantly compared to TF-IDF. This indicates that training Word2Vec on a small text corpus creates noisy vector representations, whereas TF-IDF's exact keyword matching (e.g., specific algorithm names) is more effective for this dataset size.