"""
Inference for the clean (pre-execution) XGBoost model.

Reproduces the preprocessing of notebook 05 exactly:
  1. label-encode git_branch, git_prev_commit_resolution_status, gh_lang
  2. standard-scale the 24 numeric columns (the 3 encoded ones included)
  3. leave the 2 boolean columns as booleans
  4. columns in the training order
"""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import json

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DECISION_THRESHOLD = 0.5  # XGBoost's default, as used for the reported metrics


class ArtifactError(Exception):
    """A model or preprocessing file is missing or unreadable."""


class InputError(Exception):
    """The supplied build data is invalid."""


def load_artifacts(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_path = model_dir / "xgb_clean_model.json"
    prep_path = model_dir / "preprocessing.json"
    for p in (model_path, prep_path):
        if not p.exists():
            raise ArtifactError(
                f"Missing file: {p.name}. Run 'python src\\export_artifacts.py' first."
            )
    try:
        with open(prep_path) as f:
            prep = json.load(f)
        model = XGBClassifier()
        model.load_model(str(model_path))
    except Exception as e:
        raise ArtifactError(f"Could not load model files ({type(e).__name__}).") from e

    if model.n_features_in_ != len(prep["feature_order"]):
        raise ArtifactError("Model and preprocessing files do not match (feature count).")
    return {"model": model, "prep": prep}


def prepare(df, artifacts):
    """Raw build features -> model-ready frame. Returns (X, notes)."""
    prep = artifacts["prep"]
    order = prep["feature_order"]

    missing = [c for c in order if c not in df.columns]
    if missing:
        raise InputError(f"Missing features: {missing}")
    X = df[order].copy()

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise InputError(f"Missing (NaN) values in: {bad}")

    cat_cols = prep["categorical_cols"]
    bool_cols = prep["unscaled_bool_cols"]
    num_only = [c for c in prep["numeric_cols"] if c not in cat_cols]

    # numeric columns must be real, finite numbers
    for c in num_only:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X[num_only].isna().any().any():
        bad = X[num_only].columns[X[num_only].isna().any()].tolist()
        raise InputError(f"Non-numeric values in: {bad}")
    if not np.isfinite(X[num_only].to_numpy(dtype=float)).all():
        raise InputError("Infinite values are not allowed.")

    # boolean columns must be True/False (or 1/0)
    for c in bool_cols:
        if not X[c].isin([True, False, 0, 1]).all():
            raise InputError(f"{c} must be True or False.")
        X[c] = X[c].astype(bool)

    # label-encode; a value not seen in training maps to classes[0] (notebook rule)
    notes = []
    for c in cat_cols:
        classes = prep["categorical_classes"][c]
        mapping = {v: i for i, v in enumerate(classes)}
        idx = X[c].astype(str).map(mapping)
        unseen = idx.isna()
        if unseen.any():
            notes.append(
                f"{c}: {int(unseen.sum())} value(s) not seen in training were "
                f"treated as '{classes[0]}'."
            )
        X[c] = idx.fillna(0).astype(int)

    # standard-scale the numeric columns using the training-split statistics
    num_cols = prep["numeric_cols"]
    mean = np.array(prep["scaler_mean"])
    scale = np.array(prep["scaler_scale"])
    X[num_cols] = (X[num_cols].astype("float64") - mean) / scale
    return X, notes


def predict_batch(df, artifacts):
    """Returns (failure_scores as numpy array, notes)."""
    X, notes = prepare(df, artifacts)
    try:
        scores = artifacts["model"].predict_proba(X)[:, 1]
    except Exception as e:
        raise InputError(f"Prediction failed ({type(e).__name__}).") from e
    return scores, notes


def predict_one(raw, artifacts):
    """raw: dict of the 26 features (gh_lang defaults to 'ruby')."""
    raw = dict(raw)
    raw.setdefault("gh_lang", "ruby")
    scores, notes = predict_batch(pd.DataFrame([raw]), artifacts)
    score = float(scores[0])
    return {
        "predicted_failure": score > DECISION_THRESHOLD,
        "failure_score": score,
        "notes": notes,
    }
    
import xgboost as xgb


def explain_one(raw, artifacts, top_k=8):
    """
    Per-feature contributions from XGBoost's built-in pred_contribs.
    Units are log-odds in the model's own (class-weighted) score space:
    positive pushes toward FAILURE, negative toward SUCCESS.
    """
    raw = dict(raw)
    raw.setdefault("gh_lang", "ruby")
    X, _ = prepare(pd.DataFrame([raw]), artifacts)
    try:
        contribs = artifacts["model"].get_booster().predict(
            xgb.DMatrix(X), pred_contribs=True
        )[0]
    except Exception as e:
        raise InputError(f"Explanation failed ({type(e).__name__}).") from e

    order = artifacts["prep"]["feature_order"]
    rows = [
        {"feature": f, "value": raw[f], "contribution": float(c)}
        for f, c in zip(order, contribs[:-1])
    ]
    rows.sort(key=lambda r: -abs(r["contribution"]))
    return {
        "baseline": float(contribs[-1]),   # the model's average starting point
        "total": float(contribs.sum()),    # equals the raw model output
        "top": rows[:top_k],
        "all": rows,
    }