"""
Shared pipeline code for the Streamlit app.

Everything here mirrors what the notebooks do (same cleaning, same feature
engineering, same 80:20 split with random_state=42, same chosen models), so the
app shows the same models you present in the review.

Run `python app/pipeline.py` once to train and save the models into /models.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

HOTEL_CSV = DATA_DIR / "hotel_bookings_updated_2024.csv"
PLAYERS_CSV = DATA_DIR / "players_fc24_clean.csv"

REG_PATH = MODEL_DIR / "app_regressor.joblib"
CLF_PATH = MODEL_DIR / "app_classifier.joblib"
META_PATH = MODEL_DIR / "app_metrics.json"


# --------------------------------------------------------------------------- #
# Classification: hotel booking cancellation  (mirrors classification.ipynb)
# --------------------------------------------------------------------------- #
def load_hotel_clean() -> pd.DataFrame:
    df = pd.read_csv(HOTEL_CSV)
    df["hotel_type"] = df["hotel"].str.extract(r"^(City Hotel|Resort Hotel)")
    df = df.drop(columns=["hotel"])
    df = df.drop(columns=["reservation_status", "reservation_status_date",
                          "arrival_date_year", "agent", "company", "country"])
    df = df.drop_duplicates().reset_index(drop=True)
    df["children"] = df["children"].fillna(0)
    df = df[(df["adr"] >= 0) & (df["adr"] <= 1000) & (df["adults"] <= 10)].reset_index(drop=True)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    return df


def train_classifier() -> dict:
    df = load_hotel_clean()
    X, y = df.drop(columns=["is_canceled"]), df["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)

    # Notebook's final model = default Random Forest (200 trees, no depth limit, ~94 MB).
    # For the web app we use a lighter forest (100 trees, min_samples_leaf=2, ~30 MB):
    # ROC-AUC 0.911 vs 0.912, so it fits GitHub / free hosting limits with almost no loss.
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(Xtr, y_train)
    pred, proba = rf.predict(Xte), rf.predict_proba(Xte)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "Weighted F1": f1_score(y_test, pred, average="weighted"),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }

    # UI option lists + sensible defaults, taken from the cleaned data
    options = {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols}
    ranges = {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in num_cols}
    joblib.dump({"preprocessor": preprocessor, "model": rf, "columns": X.columns.tolist(),
                 "options": options, "ranges": ranges,
                 "feature_names": preprocessor.get_feature_names_out().tolist()},
                CLF_PATH, compress=3)
    return metrics


# --------------------------------------------------------------------------- #
# Regression: FC24 player market value  (mirrors regression.ipynb)
# --------------------------------------------------------------------------- #
def load_players() -> pd.DataFrame:
    return pd.read_csv(PLAYERS_CSV)


def _regression_design(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """Engineered feature + one-hot encoding, exactly as in the notebook.

    drop_first=True is used for training (same as the notebook). For prediction we
    pass drop_first=False and then reindex to the training columns: with only one
    row, drop_first=True would wrongly drop that row's only category.
    """
    df = df.copy()
    df["athleticism_index"] = df[["pace", "movement_agility", "power_stamina",
                                  "power_strength"]].mean(axis=1)
    df[["work_rate_attack", "work_rate_defense"]] = df["work_rate"].str.split("/", expand=True)
    df = df.drop(columns=["work_rate"])
    cat_cols = ["preferred_foot", "work_rate_attack", "work_rate_defense", "position_group"]
    return pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)


def train_regressor() -> dict:
    df = load_players()
    enc = _regression_design(df)
    X = enc.drop(columns=["short_name", "player_positions", "value_eur"])
    y = np.log1p(enc["value_eur"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    Xtr, Xte = X_train.copy(), X_test.copy()
    Xtr[num_cols] = scaler.fit_transform(X_train[num_cols])
    Xte[num_cols] = scaler.transform(X_test[num_cols])

    # Final model chosen in the notebook: tuned Gradient Boosting
    gb = GradientBoostingRegressor(learning_rate=0.05, max_depth=5, n_estimators=200,
                                   random_state=RANDOM_STATE)
    gb.fit(Xtr, y_train)
    pred = gb.predict(Xte)
    metrics = {
        "R2": r2_score(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": mean_absolute_error(y_test, pred),
    }
    joblib.dump({"model": gb, "scaler": scaler, "num_cols": num_cols,
                 "feature_columns": X.columns.tolist()}, REG_PATH, compress=3)
    return metrics


def predict_player_values(bundle: dict, raw_rows: pd.DataFrame) -> np.ndarray:
    """raw_rows: rows with the same columns as players_fc24_clean.csv -> value in EUR."""
    enc = _regression_design(raw_rows, drop_first=False)
    X = enc.reindex(columns=bundle["feature_columns"], fill_value=0).astype(float)
    X[bundle["num_cols"]] = bundle["scaler"].transform(X[bundle["num_cols"]])
    return np.expm1(bundle["model"].predict(X))


# --------------------------------------------------------------------------- #
# Clustering helpers  (mirror clustering.ipynb: 5,000-player sample, 49 features)
# --------------------------------------------------------------------------- #
def clustering_data():
    df = load_players()
    X = df.select_dtypes(include=np.number).drop(columns=["value_eur", "wage_eur"])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=5000, replace=False)
    return df, X, scaler, X_scaled, idx


# --------------------------------------------------------------------------- #
def train_all():
    MODEL_DIR.mkdir(exist_ok=True)
    meta = {"classifier": train_classifier(), "regressor": train_regressor()}
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(train_all(), indent=2))
