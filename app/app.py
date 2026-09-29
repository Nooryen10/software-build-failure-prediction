"""
Streamlit demo: pre-execution build-failure prediction with the clean XGBoost model.

Demonstration layer only. All ML logic lives in src/inference.py, and every number
shown here comes from files produced by the research pipeline (models/, results/).
"""
import html
import json
import math
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_split import POST_EXECUTION_FEATURES  # noqa: E402
from src.inference import (  # noqa: E402
    ArtifactError, InputError, explain_one, load_artifacts, predict_one,
)

st.set_page_config(page_title="Build Failure Prediction", layout="wide")

# ----------------------------------------------------------------------------
# Visual identity: a quiet, paper-and-ink research look. The score scale in the
# result card is the one distinctive element; everything else stays plain.
# ----------------------------------------------------------------------------
INK, MUTED, RULE, ACCENT = "#E6EDF3", "#8B98A5", "#26334A", "#38BDF8"
FAIL, PASS, MID = "#F87171", "#34D399", "#FBBF24"
FAIL_SEG, PASS_SEG, MID_SEG = "#7F2D33", "#1E6B4E", "#7A5A14"
PANEL = "#111A2B"


CSS = """
@import url('https://fonts.googleapis.com/...');
.stApp { ... }
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
...
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: .5rem 1.1rem; }
"""
for token, value in {"__INK__": INK, "__MUTED__": MUTED, "__RULE__": RULE}.items():
    CSS = CSS.replace(token, value)
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)



# ----------------------------------------------------------------------------
# Feature labels and tooltips
# ----------------------------------------------------------------------------
FEATURE_INFO = {
    "git_branch": ("Branch name", "Git branch the build runs on. Names the model never saw in training are mapped to a fallback, and a warning is shown."),
    "git_prev_commit_resolution_status": ("Previous commit resolution", "How the previous commit was linked to an earlier build, as recorded in the dataset."),
    "gh_is_pr": ("Triggered by a pull request", "Tick if the build was triggered by a pull request."),
    "gh_pull_req_num": ("Pull request number", "Pull request number (0 if the build is not for a pull request)."),
    "gh_by_core_team_member": ("By a core team member", "Tick if the triggering commit was made by a core team member of the project."),
    "tr_build_number": ("Build number", "Sequential number of this build within its project on Travis CI."),
    "gh_repo_age": ("Repository age", "Age of the repository at build time, as recorded in the dataset."),
    "gh_repo_num_commits": ("Commits in repository", "Number of commits in the repository up to this build."),
    "gh_team_size": ("Team size", "Team size as recorded in the dataset."),
    "gh_sloc": ("Source lines of code", "Source lines of code in the project at this build."),
    "gh_num_commits_on_files_touched": ("Earlier commits on touched files", "How many earlier commits touched the files changed by this build."),
    "git_diff_src_churn": ("Source churn (lines)", "Lines of source code changed by the commits in this build."),
    "gh_diff_files_added": ("Files added", "Number of files added by this build's changes."),
    "gh_diff_files_deleted": ("Files deleted", "Number of files deleted by this build's changes."),
    "gh_diff_files_modified": ("Files modified", "Number of files modified by this build's changes."),
    "gh_diff_src_files": ("Source files changed", "Number of source-code files changed."),
    "gh_diff_doc_files": ("Documentation files changed", "Number of documentation files changed."),
    "gh_diff_other_files": ("Other files changed", "Number of changed files that are neither source nor documentation."),
    "gh_test_lines_per_kloc": ("Test lines per KLOC", "Lines of test code per 1,000 lines of source code."),
    "gh_test_cases_per_kloc": ("Test cases per KLOC", "Number of test cases per 1,000 lines of source code."),
    "gh_asserts_cases_per_kloc": ("Assertions per KLOC", "Number of assertions per 1,000 lines of source code."),
    "gh_description_complexity": ("Description complexity", "Length/complexity measure of the pull request title and description, as recorded in the dataset."),
    "gh_num_issue_comments": ("Issue comments", "Comments on the related issue or pull request discussion."),
    "gh_num_commit_comments": ("Commit comments", "Comments left on the commits in this build."),
    "gh_num_pr_comments": ("Pull request review comments", "Review comments on the pull request."),
    "gh_lang": ("Language", "Fixed to Ruby: the dataset was filtered to Ruby projects."),
}
LABEL = {f: v[0] for f, v in FEATURE_INFO.items()}
GROUPS = [
    ("Build context", ["git_branch", "git_prev_commit_resolution_status", "gh_is_pr",
                       "gh_pull_req_num", "gh_by_core_team_member", "tr_build_number"]),
    ("Repository and team", ["gh_repo_age", "gh_repo_num_commits", "gh_team_size",
                             "gh_sloc", "gh_num_commits_on_files_touched"]),
    ("Size of the change", ["git_diff_src_churn", "gh_diff_files_added", "gh_diff_files_deleted",
                            "gh_diff_files_modified", "gh_diff_src_files",
                            "gh_diff_doc_files", "gh_diff_other_files"]),
    ("Testing and discussion", ["gh_test_lines_per_kloc", "gh_test_cases_per_kloc",
                                "gh_asserts_cases_per_kloc", "gh_description_complexity",
                                "gh_num_issue_comments", "gh_num_commit_comments",
                                "gh_num_pr_comments"]),
]
FORM_FEATURES = [f for _, fs in GROUPS for f in fs]


# ----------------------------------------------------------------------------
# Loading, with plain-language errors
# ----------------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def read_json(name):
    with open(ROOT / "models" / name) as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    path = ROOT / "results" / "metrics_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


st.markdown(
    '<div class="app-head"><h1 class="app-title">Intelligent Software Build Failure Prediction Framework</h1>'
    '<p class="app-sub">Pre-Execution Machine Learning Prediction of CI Build Failures</p></div>',
    unsafe_allow_html=True,
)

try:
    ARTS = get_artifacts()
    ASSETS = read_json("demo_assets.json")
    RANGES = read_json("feature_ranges.json")
except ArtifactError as e:
    st.error(f"The model files could not be loaded. {e}")
    st.stop()
except FileNotFoundError as e:
    st.error(f"A required file is missing: {Path(str(e.filename)).name}. "
             "Run the scripts in src/ that build the model files (see the README), then reload this page.")
    st.stop()
except Exception:
    st.error("The app could not load its model files. Re-run the scripts in src/ that build them, then reload.")
    st.stop()

uncovered = set(ARTS["prep"]["feature_order"]) - set(FORM_FEATURES) - {"gh_lang"}
if uncovered:
    st.error(f"The form does not cover these model features: {sorted(uncovered)}")
    st.stop()

CLASSES = ARTS["prep"]["categorical_classes"]
KNOWN_BRANCHES = set(CLASSES["git_branch"])
FALLBACK_BRANCH = CLASSES["git_branch"][0]
METRICS = load_metrics()

# Risk tiers come from score bands measured on the held-out test set.
BANDS = ASSETS["bands"]
LOW_MAX = BANDS[1]["score_max"]
HIGH_MIN = BANDS[3]["score_max"]


def _rate(bs):
    n = sum(b["n_builds"] for b in bs)
    return sum(b["n_builds"] * b["observed_failure_rate"] for b in bs) / n


TIERS = {
    "Lower": {"rate": _rate(BANDS[:2]), "n": sum(b["n_builds"] for b in BANDS[:2]), "colour": PASS},
    "Typical": {"rate": _rate(BANDS[2:4]), "n": sum(b["n_builds"] for b in BANDS[2:4]), "colour": MID},
    "Elevated": {"rate": _rate(BANDS[4:]), "n": sum(b["n_builds"] for b in BANDS[4:]), "colour": FAIL},
}


def tier_of(score):
    if score <= LOW_MAX:
        return "Lower"
    return "Elevated" if score > HIGH_MIN else "Typical"


def clean_xgb_row():
    if METRICS is None:
        return None
    sel = METRICS[(METRICS["Model"] == "XGBoost") & (METRICS["Pipeline"] == "Clean")]
    return sel.iloc[0] if not sel.empty else None


def fmt(v):
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.6g}"
    return str(v)


def scale_html(score, tier):
    """The score scale: three tier segments, the 0.5 threshold, and a pin at the score."""
    lo, hi = LOW_MAX * 100, HIGH_MIN * 100
    pin = min(max(score * 100, 1.5), 98.5)
    seg_note = lambda t: f"{TIERS[t]['rate']:.0%} of test builds in this tier failed"  # noqa: E731
    return "".join([
        f'<div class="scale" role="img" aria-label="Failure score {score:.1%}, in the {tier} tier">',
        '<div class="track">',
        '<div class="bar">',
        f'<span style="width:{lo:.2f}%;background:{PASS_SEG}"></span>',
        f'<span style="width:{hi - lo:.2f}%;background:{MID_SEG}"></span>',
        f'<span style="width:{100 - hi:.2f}%;background:{FAIL_SEG}"></span>',
        '</div>',
        '<span class="thr" style="left:50%"></span>',
        f'<span class="pin" style="left:{pin:.2f}%"><span class="pin-val">{score:.1%}</span></span>',
        '</div>',
        '<div class="seg-labels">',
        f'<div style="width:{lo:.2f}%"><b>Lower</b><span>{seg_note("Lower")}</span></div>',
        f'<div style="width:{hi - lo:.2f}%"><b>Typical</b><span>{seg_note("Typical")}</span></div>',
        f'<div style="width:{100 - hi:.2f}%"><b>Elevated</b><span>{seg_note("Elevated")}</span></div>',
        '</div></div>',
    ])


_row = clean_xgb_row()
FALSE_ALARM_TEXT = (
    f"On the test set, only {_row['Precision']:.0%} of predicted failures were real failures."
    if _row is not None else "Precision on the test set is low (see the Model and results tab)."
)

tab_predict, tab_model, tab_about = st.tabs(["Predict", "Model and results", "About the project"])

# ============================================================================
# Predict
# ============================================================================
with tab_predict:
    st.markdown(
        '<p class="lead">Enter what is known before a build starts. The model returns a failure score, '
        'a risk tier and the features that pushed the score up or down.</p>',
        unsafe_allow_html=True,
    )

    example_names = ["Typical values (dataset medians)"] + [e["name"] for e in ASSETS["examples"]]
    choice = st.selectbox(
        "Start from",
        example_names,
        help="The examples are real builds from the held-out test set, each with its true outcome.",
    )
    idx = example_names.index(choice)
    if idx > 0:
        st.caption("This is a real build from the held-out test set. You can edit any value before predicting.")

    def manual_defaults():
        d = {f: (r["median"] if r["kind"] in ("int", "float") else False) for f, r in RANGES.items()}
        status = CLASSES["git_prev_commit_resolution_status"]
        d["git_branch"] = "master" if "master" in KNOWN_BRANCHES else FALLBACK_BRANCH
        d["git_prev_commit_resolution_status"] = "build_found" if "build_found" in status else status[0]
        d["gh_lang"] = "ruby"
        return d

    if idx == 0:
        base, truth = manual_defaults(), None
    else:
        base, truth = ASSETS["examples"][idx - 1]["features"], ASSETS["examples"][idx - 1]["true_outcome"]

    def render_input(feat):
        label, tip = FEATURE_INFO[feat]
        r, key = RANGES[feat], f"{feat}__{idx}"
        if feat == "git_branch":
            val = st.text_input(label, value=str(base[feat]), help=tip, key=key)
            if val.strip() and val.strip() not in KNOWN_BRANCHES:
                st.caption("Not seen in training. The model will use a fallback branch.")
            return val
        if feat == "git_prev_commit_resolution_status":
            opts = CLASSES[feat]
            cur = base[feat] if base[feat] in opts else opts[0]
            return st.selectbox(label, opts, index=opts.index(cur), help=tip, key=key)
        if r["kind"] == "bool":
            return st.checkbox(label, value=bool(base[feat]), help=tip, key=key)
        if r["kind"] == "int":
            return st.number_input(label, value=int(base[feat]),
                                   min_value=0 if r["min"] >= 0 else None,
                                   step=1, format="%d", help=tip, key=key)
        return st.number_input(label, value=float(base[feat]),
                               min_value=0.0 if r["min"] >= 0 else None,
                               step=1.0 if r["max"] >= 100 else 0.01,
                               format="%g", help=tip, key=key)

    values = {}
    for group_name, feats in GROUPS:
        with st.container(border=True):
            st.markdown(f"**{group_name}**")
            cols = st.columns(3)
            for i, f in enumerate(feats):
                with cols[i % 3]:
                    values[f] = render_input(f)
    st.caption("Language is fixed to Ruby, because the dataset was filtered to Ruby projects.")

    def _close(a, b):
        nums = (int, float)
        if isinstance(a, nums) and isinstance(b, nums) and not isinstance(a, bool) and not isinstance(b, bool):
            return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6)
        return a == b

    def same_inputs(a, b):
        return all(_close(a[f], b[f]) for f in FORM_FEATURES)

    def validate(vals):
        errors, warns = [], []
        for f, v in vals.items():
            label, r = LABEL[f], RANGES[f]
            if f == "git_branch":
                if not str(v).strip():
                    errors.append(f"{label}: enter a branch name.")
            elif r["kind"] in ("int", "float"):
                if v is None:
                    errors.append(f"{label}: enter a number.")
                elif not math.isfinite(float(v)):
                    errors.append(f"{label}: the value must be a finite number.")
                elif v < r["min"] or v > r["max"]:
                    warns.append(f"{label} = {fmt(v)} is outside the range seen in the dataset "
                                 f"({fmt(r['min'])} to {fmt(r['max'])}), so the prediction may be less reliable.")
        return errors, warns

    def payload_from(vals):
        p = {}
        for f, v in vals.items():
            kind = RANGES[f]["kind"]
            if f == "git_branch":
                p[f] = str(v).strip()
            elif kind == "int":
                p[f] = int(v)
            elif kind == "float":
                p[f] = float(v)
            elif kind == "bool":
                p[f] = bool(v)
            else:
                p[f] = v
        return p

    st.write("")
    clicked = st.button("Predict Build Failure", type="primary")
    if clicked:
        errors, warns = validate(values)
        if errors:
            st.error("The prediction was not run. Fix the following and try again:\n\n"
                     + "\n".join(f"- {e}" for e in errors))
        else:
            payload = payload_from(values)
            try:
                result = predict_one(payload, ARTS)
                expl = explain_one(payload, ARTS, top_k=8)
            except (InputError, ArtifactError) as e:
                st.error(f"The prediction could not be made. {e}")
            except Exception:
                st.error("Something unexpected went wrong while predicting. Check the inputs and try again.")
            else:
                st.session_state["last"] = {"payload": payload, "result": result,
                                            "expl": expl, "warns": warns}

    def show_result(last, stale):
        payload, result, expl = last["payload"], last["result"], last["expl"]
        score = result["failure_score"]
        tier = tier_of(score)
        failed = result["predicted_failure"]
        edge = FAIL if failed else PASS

        if stale:
            st.warning("The inputs changed after this prediction. Click Predict Build Failure to update it.")
        for w in last["warns"]:
            st.warning(w)

        row = clean_xgb_row()
        precision_line = ""
        if row is not None:
            precision_line = (f" When this model predicts failure, about {row['Precision']:.0%} of those builds "
                              "really fail, so many warnings are false alarms.")
        card = "".join([
            f'<div class="result{" stale" if stale else ""}" style="--edge:{edge}">',
            f'<p class="verdict">The model predicts this build will {"fail" if failed else "pass"}.</p>',
            '<div class="stats">',
            f'<div class="stat"><div class="k">Prediction</div><div class="v">{"FAILURE" if failed else "SUCCESS"}</div></div>',
            f'<div class="stat"><div class="k">Failure score</div><div class="v">{score:.1%}</div></div>',
            f'<div class="stat"><div class="k">Risk tier</div><div class="v" style="color:{TIERS[tier]["colour"]}">{tier}</div></div>',
            '</div>',
            scale_html(score, tier),
            f'<p class="note">In the held-out test set, builds scoring in the {tier} tier failed '
            f'{TIERS[tier]["rate"]:.0%} of the time. Overall, {ASSETS["test_failure_rate"]:.1%} of test builds failed.'
            f'{html.escape(precision_line)}</p>',
            '</div>',
        ])
        st.markdown(card, unsafe_allow_html=True)
        st.caption("The score is the model's output, not a calibrated probability: the model was trained with "
                   "class weighting, so it should not be read as a literal chance of failure. The dashed line "
                   "marks the 0.5 threshold that separates FAILURE from SUCCESS. Tiers are relative and come "
                   "from the measured test-set results shown above.")
        if abs(score - 0.5) < 0.05:
            st.warning("The score is close to the 0.5 threshold, so treat this prediction as borderline.")

        unseen = payload["git_branch"] not in KNOWN_BRANCHES
        if unseen:
            st.warning(f"The branch name '{payload['git_branch']}' was not seen in training, so the model treated it "
                       f"as '{FALLBACK_BRANCH}'. Branch is the model's most important feature, so this prediction "
                       "depends on that fallback.")
        if truth is not None and idx > 0 and same_inputs(payload, base):
            correct = failed == (truth == "FAILED")
            st.info(f"This is a real test-set build. Its actual outcome was **{truth}**, so the model's "
                    f"prediction was **{'correct' if correct else 'incorrect'}**.")

        st.markdown("#### Why the model produced this score")
        rows = []
        for r in expl["top"]:
            shown = fmt(r["value"])
            if r["feature"] == "git_branch" and unseen:
                shown += " (unseen; fallback used)"
            rows.append({"Feature": LABEL[r["feature"]], "Your value": shown,
                         "Contribution": round(r["contribution"], 3),
                         "Pushes toward": "Failure" if r["contribution"] > 0 else "Success"})
        contrib = pd.DataFrame(rows)
        bars = alt.Chart(contrib).mark_bar(size=16).encode(
            x=alt.X("Contribution:Q", title="Pushes toward success (left) or failure (right)"),
            y=alt.Y("Feature:N", sort=list(contrib["Feature"]), title=None, axis=alt.Axis(labelLimit=280)),
            color=alt.Color("Pushes toward:N", legend=None,
                            scale=alt.Scale(domain=["Failure", "Success"], range=[FAIL, PASS])),
            tooltip=["Feature", "Your value", "Contribution", "Pushes toward"],
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK).encode(x="x:Q")
        st.altair_chart((bars + zero).properties(height=30 * len(contrib) + 24), width="stretch")
        with st.expander("Show the exact values"):
            st.dataframe(contrib, hide_index=True, width="stretch")
        st.caption("Contributions come from the trained XGBoost model's built-in feature contributions, in the "
                   "units of its raw (log-odds) output. Together with the model's baseline they add up exactly "
                   "to the score above. They show how the model reached its score, not why a build really fails.")

    last = st.session_state.get("last")
    if last is not None:
        show_result(last, stale=not same_inputs(last["payload"], values))

# ============================================================================
# Model and results
# ============================================================================
with tab_model:
    st.markdown(
        f"The demo uses the **clean XGBoost** model: {len(ARTS['prep']['feature_order'])} pre-execution features, "
        f"trained on TravisTorrent Ruby builds and evaluated on the {ASSETS['n_test_builds']:,} most recent builds "
        "(a chronological 80/20 split, decision threshold 0.5). The prediction model was evaluated using the "
        "experimental setup described in the research project."
    )

    if METRICS is None:
        st.info("results/metrics_comparison.csv was not found, so the metrics below are unavailable.")
    else:
        M = METRICS.rename(columns={"AUC-ROC": "AUC"})
        st.markdown("#### Clean XGBoost against its leaky counterpart")
        cx = M[(M["Model"] == "XGBoost") & (M["Pipeline"] == "Clean")].iloc[0]
        lx = M[(M["Model"] == "XGBoost") & (M["Pipeline"] == "Leaky")].iloc[0]
        cols = st.columns(4)
        for c, (name, key) in zip(cols, [("Precision", "Precision"), ("Recall", "Recall"),
                                         ("F1", "F1"), ("AUC-ROC", "AUC")]):
            c.metric(name, f"{cx[key]:.3f}", f"{cx[key] - lx[key]:+.3f} vs leaky", delta_color="off")

        st.markdown("#### Every model, both pipelines")
        pipe_scale = alt.Scale(domain=["Leaky", "Clean"], range=["#98A3AE", ACCENT])

        def metric_chart(metric, title):
            base_c = alt.Chart(M).encode(
                x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="Pipeline:N",
                y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Pipeline:N", scale=pipe_scale, legend=alt.Legend(title=None, orient="top")),
            )
            labels = base_c.mark_text(dy=-7, fontSize=11).encode(text=alt.Text(f"{metric}:Q", format=".3f"))
            return (base_c.mark_bar() + labels).properties(height=280)

        c1, c2 = st.columns(2)
        c1.altair_chart(metric_chart("F1", "F1 score"), width="stretch")
        c2.altair_chart(metric_chart("AUC", "AUC-ROC"), width="stretch")

        table = []
        for name in M["Model"].unique():
            lk = M[(M["Model"] == name) & (M["Pipeline"] == "Leaky")].iloc[0]
            cl = M[(M["Model"] == name) & (M["Pipeline"] == "Clean")].iloc[0]
            table.append({"Model": name, "F1 (leaky)": f"{lk['F1']:.3f}", "F1 (clean)": f"{cl['F1']:.3f}",
                          "F1 drop": f"{(lk['F1'] - cl['F1']) / lk['F1']:.1%}",
                          "AUC (leaky)": f"{lk['AUC']:.3f}", "AUC (clean)": f"{cl['AUC']:.3f}"})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Metrics come from results/metrics_comparison.csv, written by notebooks 04 and 05. "
                   "The leaky pipeline also uses features that only exist after the build has started.")

    st.markdown("#### Where the risk tiers come from")
    st.markdown(
        "The test builds were split into five equal groups by model score. The chart shows how often builds in "
        "each group actually failed. Failure becomes more common as the score rises, which is why the tiers are "
        "useful as a relative ranking, but even the highest group fails well under half of the time."
    )
    band_df = pd.DataFrame([{
        "Score range": f"{b['score_min']:.2f} to {b['score_max']:.2f}",
        "Observed failure rate": b["observed_failure_rate"],
    } for b in BANDS])
    top = max(band_df["Observed failure rate"].max(), ASSETS["test_failure_rate"]) * 1.25
    bar = alt.Chart(band_df).mark_bar(color=ACCENT).encode(
        x=alt.X("Score range:N", sort=list(band_df["Score range"]), axis=alt.Axis(labelAngle=0),
                title="Model score range (fifths of the test set)"),
        y=alt.Y("Observed failure rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, top])),
    )
    txt = bar.mark_text(dy=-7, fontSize=11).encode(text=alt.Text("Observed failure rate:Q", format=".0%"))
    avg = alt.Chart(pd.DataFrame({"y": [ASSETS["test_failure_rate"]]})).mark_rule(
        strokeDash=[4, 3], color=MUTED).encode(y="y:Q")
    st.altair_chart((bar + txt + avg).properties(height=280), width="stretch")
    st.caption(f"The dashed line is the overall test failure rate ({ASSETS['test_failure_rate']:.1%}). "
               "Tiers: Lower is the two lowest groups, Typical the two middle groups, Elevated the highest group.")
    tier_table = pd.DataFrame([
        {"Tier": "Lower", "Model score": f"up to {LOW_MAX:.2f}",
         "Test builds": f"{TIERS['Lower']['n']:,}", "Observed failure rate": f"{TIERS['Lower']['rate']:.1%}"},
        {"Tier": "Typical", "Model score": f"{LOW_MAX:.2f} to {HIGH_MIN:.2f}",
         "Test builds": f"{TIERS['Typical']['n']:,}", "Observed failure rate": f"{TIERS['Typical']['rate']:.1%}"},
        {"Tier": "Elevated", "Model score": f"above {HIGH_MIN:.2f}",
         "Test builds": f"{TIERS['Elevated']['n']:,}", "Observed failure rate": f"{TIERS['Elevated']['rate']:.1%}"},
    ])
    st.dataframe(tier_table, hide_index=True, width="stretch")

    st.markdown("#### What the model relies on overall")
    imp = pd.DataFrame(ASSETS["feature_importance"][:10])
    imp["Feature"] = imp["feature"].map(lambda f: LABEL.get(f, f))
    imp_chart = alt.Chart(imp).mark_bar(color=ACCENT, size=16).encode(
        x=alt.X("importance:Q", title="Importance"),
        y=alt.Y("Feature:N", sort=list(imp["Feature"]), title=None, axis=alt.Axis(labelLimit=280)),
        tooltip=["Feature", alt.Tooltip("importance:Q", format=".3f")],
    ).properties(height=300)
    st.altair_chart(imp_chart, width="stretch")
    st.caption("XGBoost's built-in feature importance for the clean model (top 10). It describes the model "
               "overall, not any single build.")

    with st.expander("Post-execution features excluded from the clean pipeline"):
        st.write(f"These {len(POST_EXECUTION_FEATURES)} features only exist after or during a build, so the "
                 "clean pipeline does not use them:")
        st.write(", ".join(POST_EXECUTION_FEATURES))

# ============================================================================
# About
# ============================================================================
with tab_about:
    st.markdown(
        """
#### What this demo does
Given information available **before** a software build starts, it estimates whether the build is likely to
fail or succeed.

#### Why prediction has to happen before the build
A warning is only useful while there is still time to act on it, for example by reviewing a risky change first.
Once the build has run, the outcome is already known.

#### Data leakage
Some earlier studies report very high accuracy by using information that only exists after a build has run,
such as test results or build duration. Such a model looks excellent but could never be used at prediction time.
This project compares a **leaky** pipeline with a **clean** pipeline that uses only pre-execution information,
and measures how much performance drops. This app uses the clean pipeline.

#### How the demo works
1. You enter the engineered build features, or load a real build from the test set.
2. The inputs are checked for missing or invalid values.
3. Text features are encoded and numbers are scaled exactly as they were during training.
4. The trained clean XGBoost model produces a failure score.
5. The app shows the score, a risk tier taken from measured test-set results, and the features that drove the score.

The research model operates on engineered pre-execution features from the TravisTorrent dataset, so the demo
accepts those features directly. It cannot derive them from a repository URL.

#### Limitations
- **The score is not a calibrated probability.** The model was trained with class weighting.
- **Most failure warnings are false alarms.** __FALSE_ALARM__
- **Unseen branch names.** Roughly a quarter of the test builds ran on branch names the model never saw in training. They are mapped to a fallback branch, and this is already reflected in the reported results.
- **One dataset, one split.** Results come from TravisTorrent Ruby projects and a single chronological split. They may not carry over to other languages or CI systems.
- **Explanations describe the model, not causes.** A feature that pushes the score up is not necessarily why a build fails.

#### Research contribution and demonstration
The research contribution is the comparison of leaky and clean pipelines in the notebooks. This application
is a demonstration layer that shows how the clean model could be used. It does not change or add to the
research results.
        """.replace("__FALSE_ALARM__", FALSE_ALARM_TEXT)
    )

st.divider()
st.caption("Demonstration layer for a research project. Not the primary research contribution.")