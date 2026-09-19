"""Records each feature's type, observed min/max and median from df_clean.csv."""
import json

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

with open("models/preprocessing.json") as f:
    order = json.load(f)["feature_order"]

df = pd.read_csv("data/processed/df_clean.csv")
out = {}
for f in order:
    s = df[f]
    if is_bool_dtype(s):
        out[f] = {"kind": "bool"}
    elif is_integer_dtype(s):
        out[f] = {"kind": "int", "min": int(s.min()), "max": int(s.max()),
                  "median": int(round(s.median()))}
    elif is_float_dtype(s):
        out[f] = {"kind": "float", "min": float(s.min()), "max": float(s.max()),
                  "median": float(s.median())}
    else:
        out[f] = {"kind": "cat"}

with open("models/feature_ranges.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"{'feature':<36}{'kind':<7}{'min':>14}{'median':>14}{'max':>16}")
for f, r in out.items():
    if r["kind"] in ("int", "float"):
        print(f"{f:<36}{r['kind']:<7}{r['min']:>14.6g}{r['median']:>14.6g}{r['max']:>16.6g}")
    else:
        print(f"{f:<36}{r['kind']:<7}")
print("\nSaved models/feature_ranges.json")