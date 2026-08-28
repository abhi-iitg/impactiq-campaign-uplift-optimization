"""
Train the winning uplift model (X-learner) and save it for the Streamlit app.
Run once after the notebooks:  python src/train_model.py
Produces:  models/xlearner.pkl  and  models/feature_cols.json
"""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_models import XLearner

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)

FEATURES = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
OUTCOME = "visit"   # modeling proxy (dense)


def main():
    df = pd.read_csv(PROC / "hillstrom_clean.csv")
    y = df[OUTCOME].astype(int).values
    w = df["treatment"].astype(int).values
    X = pd.get_dummies(df[FEATURES], drop_first=True)

    X_tr, _, w_tr, _, y_tr, _ = train_test_split(
        X, w, y, test_size=0.30, random_state=42, stratify=w
    )
    model = XLearner().fit(X_tr, w_tr, y_tr)

    with open(MODELS / "xlearner.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS / "feature_cols.json", "w") as f:
        json.dump(list(X.columns), f)

    # also save category options for the app's dropdowns
    cats = {c: sorted(df[c].dropna().unique().tolist())
            for c in ["zip_code", "channel"]}
    with open(MODELS / "categories.json", "w") as f:
        json.dump(cats, f)

    print(f"Saved model + {len(X.columns)} feature columns -> {MODELS}")


if __name__ == "__main__":
    main()
