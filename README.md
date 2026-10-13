# Machine Learning Capstone - 23CSE301 (2026-27)

End-to-end ML pipelines for three tracks, plus an interactive Streamlit app.

**Team:** _<names and roll numbers>_

| Track | Problem | Dataset | Final model |
|---|---|---|---|
| Regression | Predict a player's market value (log-transformed `value_eur`) | EA FC 24 players (18,250 rows) | Tuned Gradient Boosting - test R² 0.9978 |
| Classification | Predict whether a hotel booking is cancelled (`is_canceled`) | Hotel bookings 2024 (119,390 rows) | Random Forest - ROC-AUC 0.912, accuracy 0.842 |
| Clustering | Group players by attributes without labels | EA FC 24 players (5,000-player sample) | K-Means, k=4 - silhouette 0.228 |

_<Add the 10-model comparison tables here, or link to the notebooks.>_

## Repository structure

```
README.md
requirements.txt
data/        hotel_bookings_updated_2024.csv, players_fc24_clean.csv
notebooks/   regression.ipynb, classification.ipynb, clustering.ipynb
models/      saved models (created by app/pipeline.py) and result tables
app/         streamlit_app.py (the GUI), pipeline.py (shared code + training)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

1. **Notebooks:** open `notebooks/` in Jupyter and run each notebook top to bottom.
2. **App:**
   ```bash
   python app/pipeline.py             # trains and saves the app models into models/ (about 2-3 min)
   streamlit run app/streamlit_app.py
   ```
   If `models/` is empty, the app trains them automatically on first launch.

## App

| Page | What it does |
|---|---|
| Hotel cancellation | Enter booking details, get the cancellation probability and risk level |
| Player value | Pick a player, edit attributes, see the re-estimated market value |
| Player clusters | Choose K-Means / Agglomerative and k, see metrics, PCA plot, profiles, elbow curve, dendrogram |

The app's classifier is a lighter Random Forest (100 trees, `min_samples_leaf=2`, 30 MB) than the
notebook's 200-tree model (94 MB) so it fits GitHub and free hosting limits. Test ROC-AUC is 0.911 vs 0.912.

**Live app:** _<public URL once deployed>_

## Data

- `players_fc24_clean.csv` is the cleaned FC 24 subset produced in `regression.ipynb`.
  The raw file (`male_players.csv`, ~96 MB, FIFA 15 to FC 24) is not committed; _<add download link>_.
- `hotel_bookings_updated_2024.csv` - hotel booking records.

## Notes and AI assistance

_<Cite any AI assistance used for code scaffolding here, as required by the course guidelines.>_
