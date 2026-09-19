"""Proves src/inference.py reproduces the reported clean-XGBoost results."""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.inference import load_artifacts, predict_batch, predict_one

df = pd.read_csv("data/processed/df_clean.csv").sort_values("gh_build_started_at")
test = df.iloc[int(len(df) * 0.8):]
raw = test.drop(columns=["target", "gh_build_started_at"])
y = test["target"]

arts = load_artifacts()
scores, notes = predict_batch(raw, arts)
pred = (scores > 0.5).astype(int)
print("Notes:", notes if notes else "none")

ref = pd.read_csv("results/metrics_comparison.csv")
ref = ref[(ref["Model"] == "XGBoost") & (ref["Pipeline"] == "Clean")].iloc[0]
got = {
    "Precision": precision_score(y, pred),
    "Recall": recall_score(y, pred),
    "F1": f1_score(y, pred),
    "AUC-ROC": roc_auc_score(y, scores),
}
ok = True
print(f"{'Metric':<10} {'inference.py':>12} {'Reported':>9}")
for k, v in got.items():
    ok &= abs(v - ref[k]) < 0.001
    print(f"{k:<10} {v:>12.4f} {ref[k]:>9.4f}")

# single-row path must agree with the batch path
sample = raw.sample(300, random_state=0)
batch_scores = scores[[raw.index.get_loc(i) for i in sample.index]]
one_scores = np.array([predict_one(r, arts)["failure_score"]
                       for r in sample.to_dict("records")])
diff = np.abs(batch_scores - one_scores).max()
print("Max score difference, single-row vs batch (300 builds):", diff)
ok &= diff < 1e-6

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)