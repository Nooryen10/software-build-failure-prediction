"""Checks that explanations are consistent with the model's own score."""
import json
import math
import sys

from src.inference import explain_one, load_artifacts, predict_one

arts = load_artifacts()
examples = json.load(open("models/demo_assets.json"))["examples"]

ok = True
for ex in examples:
    res = predict_one(ex["features"], arts)
    exp = explain_one(ex["features"], arts)
    from_contribs = 1 / (1 + math.exp(-exp["total"]))
    diff = abs(from_contribs - res["failure_score"])
    ok &= diff < 1e-4
    print(f"{ex['name']}")
    print(f"   true={ex['true_outcome']}  score={res['failure_score']:.4f}  "
          f"score from contributions={from_contribs:.4f}  diff={diff:.6f}")
    for r in exp["top"][:3]:
        sign = "toward FAILURE" if r["contribution"] > 0 else "toward SUCCESS"
        print(f"     {r['feature']} = {r['value']}  ({r['contribution']:+.3f}, {sign})")

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)