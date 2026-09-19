"""
One-time script. Rebuilds the preprocessing objects from notebook 05
(LabelEncoders + StandardScaler, fitted on the training split only) and
verifies that the saved clean XGBoost model reproduces the reported metrics.
Nothing is trained here. Artifacts are written only if verification passes.
"""
import json
import os
import shutil
import sys

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "data/processed/df_clean.csv"
MODEL_PATH = "results/xgb_clean_model.json"
NAMES_PATH = "results/clean_feature_names.pkl"
METRICS_PATH = "results/metrics_comparison.csv"
OUT_DIR = "models"

# Named explicitly (not auto-detected) so a newer pandas cannot silently skip them.
CATEGORICAL_COLS = ["git_branch", "git_prev_commit_resolution_status", "gh_lang"]
TRAIN_FRAC = 0.8   # same chronological 80/20 split as notebook 05
TOLERANCE = 0.001


def main():
    feature_names = list(joblib.load(NAMES_PATH))

    # 1. Same chronological split as notebook 05
    df = pd.read_csv(DATA_PATH).sort_values("gh_build_started_at")
    split_idx = int(len(df) * TRAIN_FRAC)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print("Train:", train_df.shape, "Test:", test_df.shape)

    drop = ["target", "gh_build_started_at"]
    X_train = train_df.drop(columns=drop).copy()
    X_test = test_df.drop(columns=drop).copy()
    y_test = test_df["target"]

    # 2. Label-encode categoricals, fitted on the training split only
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        known = set(le.classes_)
        X_test[col] = X_test[col].astype(str).map(
            lambda v: v if v in known else le.classes_[0]
        )
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    # 3. Standard-scale every numeric column (as in notebook 05)
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    bool_cols = [c for c in X_train.columns if c not in numeric_cols]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    print("Numeric columns scaled:", len(numeric_cols))
    print("Columns left unscaled:", bool_cols)

    if list(X_train.columns) != feature_names:
        sys.exit("STOP: column order differs from clean_feature_names.pkl")

    # 4. Score the saved model on the test split
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    X_test = X_test[feature_names]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    got = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }

    # 5. Compare with the numbers already stored in results/metrics_comparison.csv
    ref_all = pd.read_csv(METRICS_PATH)
    ref = ref_all[(ref_all["Model"] == "XGBoost") & (ref_all["Pipeline"] == "Clean")].iloc[0]
    all_ok = True
    print(f"\n{'Metric':<10} {'Rebuilt':>9} {'Reported':>9} {'Diff':>9}")
    for k, v in got.items():
        diff = abs(v - ref[k])
        all_ok &= diff < TOLERANCE
        print(f"{k:<10} {v:>9.4f} {ref[k]:>9.4f} {diff:>9.5f}")

    if not all_ok:
        sys.exit("\nSTOP: metrics do not match. Nothing was saved.")
    print("\nVERIFIED: the rebuilt preprocessing reproduces the reported results.")

    # 6. Save portable artifacts (JSON, no pickles)
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copyfile(MODEL_PATH, os.path.join(OUT_DIR, "xgb_clean_model.json"))
    artifacts = {
        "feature_order": feature_names,
        "categorical_cols": CATEGORICAL_COLS,
        "categorical_classes": {c: encoders[c].classes_.tolist() for c in CATEGORICAL_COLS},
        "unseen_category_rule": "map to classes[0], as in notebook 05",
        "numeric_cols": numeric_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "unscaled_bool_cols": bool_cols,
    }
    with open(os.path.join(OUT_DIR, "preprocessing.json"), "w") as f:
        json.dump(artifacts, f)
    print("Saved models/xgb_clean_model.json and models/preprocessing.json")


if __name__ == "__main__":
    main()