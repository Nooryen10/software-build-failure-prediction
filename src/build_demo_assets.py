"""Builds models/demo_assets.json from the held-out test split. Nothing is trained."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import DECISION_THRESHOLD, load_artifacts, predict_batch

df = pd.read_csv("data/processed/df_clean.csv").sort_values("gh_build_started_at")
test = df.iloc[int(len(df) * 0.8):]
raw = test.drop(columns=["target", "gh_build_started_at"])
y = test["target"].to_numpy()

arts = load_artifacts()
scores, _ = predict_batch(raw, arts)
pred = scores > DECISION_THRESHOLD

# 1. Score bands: quintiles of the test-set score, with the observed failure rate in each
edges = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
band = np.digitize(scores, edges)
bands = []
print(f"Test builds: {len(y)}, overall failure rate: {y.mean():.3f}\n")
print(f"{'Band':<5} {'Score range':<18} {'Builds':>7} {'Observed failure rate':>22}")
for b in range(5):
    m = band == b
    bands.append({
        "band": b + 1,
        "score_min": float(scores[m].min()),
        "score_max": float(scores[m].max()),
        "n_builds": int(m.sum()),
        "observed_failure_rate": float(y[m].mean()),
    })
    print(f"{b + 1:<5} {scores[m].min():.3f} - {scores[m].max():.3f}     "
          f"{m.sum():>7} {y[m].mean():>22.3f}")

# 2. Real example builds from the test split (fixed seed, nothing invented)
known_branches = set(arts["prep"]["categorical_classes"]["git_branch"])
unseen_branch = ~raw["git_branch"].astype(str).isin(known_branches).to_numpy()
cases = {
    "Failed build, predicted to fail (correct)": (y == 1) & pred,
    "Passed build, predicted to pass (correct)": (y == 0) & ~pred,
    "Failed build, missed by the model": (y == 1) & ~pred,
    "Passed build, false alarm": (y == 0) & pred,
    "Build on a branch not seen in training": unseen_branch,
}
rng = np.random.default_rng(0)
examples = []
for name, mask in cases.items():
    i = int(rng.choice(np.flatnonzero(mask)))
    record = json.loads(raw.iloc[[i]].to_json(orient="records"))[0]
    examples.append({
        "name": name,
        "true_outcome": "FAILED" if y[i] == 1 else "PASSED",
        "score_at_export": float(scores[i]),
        "features": record,
    })

# 3. Feature importance straight from the trained model
model = arts["model"]
names = arts["prep"]["feature_order"]
imp = sorted(zip(names, model.feature_importances_.tolist()), key=lambda t: -t[1])

out = {
    "n_test_builds": int(len(y)),
    "test_failure_rate": float(y.mean()),
    "bands": bands,
    "examples": examples,
    "feature_importance": [{"feature": n, "importance": v} for n, v in imp],
}
Path("models/demo_assets.json").write_text(json.dumps(out, indent=2))
print("\nSaved models/demo_assets.json")
print("Top 5 features:", [n for n, _ in imp[:5]])
for e in examples:
    print(f"  {e['name']}: true={e['true_outcome']}, score={e['score_at_export']:.3f}")