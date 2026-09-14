"""
69_batch_review.py

Purpose:
--------
Every real bug found in this whole project (the Cuas 2-pitch DROP
case, the Ben Rowen extrapolation bug) was found by scrutinizing
real output closely -- but always from the SAME small, recurring
set of demo pitchers, never a broad, unsupervised sweep. Before
packaging, this runs the FULLY integrated engine across a large,
diverse sample of real pitchers and scans for anything that looks
broken, not just confirms it runs without crashing.

Robust to individual failures -- one pitcher's error doesn't kill
the batch, it's logged and the sweep continues, since finding WHERE
things break is the whole point of this exercise.

Flags checked per pitcher:
- Crashes/exceptions
- No viable candidates found at all
- Extreme predicted_change values (beyond a plausible real range)
- exceeds_historical_precedent firing
- Impossible predicted_outcome values (outside a real xwOBA range)
- How often NO_CHANGE ranks #1 (worth knowing the base rate)

Output:
-------
Printed summary counts per flag category, plus the specific worst
cases for manual review. No file saved -- this is a review pass,
not a deliverable.
"""

import importlib.util
import sys
import traceback

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT 45 AND 60
# ============================================================

print("Loading recommendation engine (45) and Stage 1 integration (60)...")

rec_spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine", "45_similarity_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(rec_spec)
sys.modules["similarity_recommendation_engine"] = rec_engine
rec_spec.loader.exec_module(rec_engine)

recommend_arsenal_changes = rec_engine.recommend_arsenal_changes
similarity_features = rec_engine.similarity_features
SIMILARITY_FEATURE_COLS = rec_engine.SIMILARITY_FEATURE_COLS

stage60_spec = importlib.util.spec_from_file_location(
    "stage1_all_change_types", "60_stage1_all_change_types.py"
)
stage60 = importlib.util.module_from_spec(stage60_spec)
sys.modules["stage1_all_change_types"] = stage60
stage60_spec.loader.exec_module(stage60)

predict_arsenal_change_with_stage1_all_types = stage60.predict_arsenal_change_with_stage1_all_types


# ============================================================
# SETTINGS
# ============================================================

N_PITCHERS = 75  # broad, diverse, unsupervised -- not the same
                   # recurring demo cases
RANDOM_SEED = 7  # deliberately different from every seed used
                   # elsewhere in this project

# real-world plausible bounds for xwoba_against -- outside this,
# something is almost certainly wrong, not just an unusual real case
PLAUSIBLE_OUTCOME_MIN = 0.10
PLAUSIBLE_OUTCOME_MAX = 0.55
PLAUSIBLE_CHANGE_MAX_ABS = 0.15  # a season-over-season swing bigger
                                    # than this from ONE arsenal
                                    # change is implausible


# ============================================================
# SAMPLE A BROAD, DIVERSE SET OF REAL PITCHERS
# ============================================================

print(f"\nSampling {N_PITCHERS} pitchers (broad, random, unsupervised)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
target_sample = complete_profiles.sample(n=min(N_PITCHERS, len(complete_profiles)), random_state=RANDOM_SEED)

print(f"Sampled {len(target_sample):,} pitcher-seasons")


# ============================================================
# RUN THE FULL ENGINE, LOGGING ISSUES ROBUSTLY
# ============================================================

print("\nRunning the fully-integrated engine across the full sample...")

flags = {
    "crashed": [],
    "no_candidates": [],
    "extreme_predicted_change": [],
    "exceeds_historical_precedent": [],
    "impossible_outcome": [],
    "no_change_ranked_first": [],
}
n_successful = 0

for _, target_row in target_sample.iterrows():
    player_id = target_row["player_id"]
    season = target_row["season"]
    label = f"{target_row.get('player_name', player_id)}_{season}"

    try:
        recommendations, result_objects = recommend_arsenal_changes(
            player_id, season, outcome_metric="xwoba_against",
            add_prediction_fn=predict_arsenal_change_with_stage1_all_types,
            other_change_prediction_fn=predict_arsenal_change_with_stage1_all_types
        )
    except Exception as e:
        flags["crashed"].append({"target": label, "error": str(e), "traceback": traceback.format_exc()})
        continue

    n_successful += 1

    if len(recommendations) == 0:
        flags["no_candidates"].append({"target": label})
        continue

    extreme_rows = recommendations[recommendations["predicted_change"].abs() > PLAUSIBLE_CHANGE_MAX_ABS]
    for _, row in extreme_rows.iterrows():
        flags["extreme_predicted_change"].append({
            "target": label, "change_type": row["change_type"], "pitch_type": row["pitch_type"],
            "predicted_change": row["predicted_change"], "n_historical_events": row.get("n_historical_events"),
        })

    if "exceeds_historical_precedent" in recommendations.columns:
        exceed_rows = recommendations[recommendations["exceeds_historical_precedent"] == True]
        for _, row in exceed_rows.iterrows():
            flags["exceeds_historical_precedent"].append({
                "target": label, "change_type": row["change_type"], "pitch_type": row["pitch_type"],
            })

    impossible_rows = recommendations[
        (recommendations["predicted_outcome"] < PLAUSIBLE_OUTCOME_MIN)
        | (recommendations["predicted_outcome"] > PLAUSIBLE_OUTCOME_MAX)
    ]
    for _, row in impossible_rows.iterrows():
        flags["impossible_outcome"].append({
            "target": label, "change_type": row["change_type"], "pitch_type": row["pitch_type"],
            "predicted_outcome": row["predicted_outcome"],
        })

    if recommendations.iloc[0]["change_type"] == "NO_CHANGE":
        flags["no_change_ranked_first"].append({"target": label})


print(f"\n{n_successful:,} of {len(target_sample):,} pitchers processed without crashing")


# ============================================================
# SUMMARY
# ============================================================

print("\n\n==============================")
print("Batch review summary")
print("==============================")

print(f"\nCrashed: {len(flags['crashed'])} of {len(target_sample)}")
print(f"No candidates found: {len(flags['no_candidates'])}")
print(f"Extreme predicted_change (|change| > {PLAUSIBLE_CHANGE_MAX_ABS}): {len(flags['extreme_predicted_change'])} rows")
print(f"Exceeds historical precedent flagged: {len(flags['exceeds_historical_precedent'])} rows")
print(f"Impossible predicted_outcome (outside [{PLAUSIBLE_OUTCOME_MIN}, {PLAUSIBLE_OUTCOME_MAX}]): {len(flags['impossible_outcome'])} rows")
print(f"NO_CHANGE ranked #1: {len(flags['no_change_ranked_first'])} of {n_successful} successful pitchers "
      f"({len(flags['no_change_ranked_first'])/n_successful:.1%})" if n_successful > 0 else "")

if flags["crashed"]:
    print("\n\n--- CRASHES (highest priority to investigate) ---")
    for c in flags["crashed"][:5]:
        print(f"\n{c['target']}: {c['error']}")

if flags["impossible_outcome"]:
    print("\n\n--- IMPOSSIBLE OUTCOME VALUES ---")
    print(pd.DataFrame(flags["impossible_outcome"]).to_string(index=False))

if flags["extreme_predicted_change"]:
    print("\n\n--- EXTREME PREDICTED CHANGES (worth spot-checking) ---")
    extreme_df = pd.DataFrame(flags["extreme_predicted_change"]).sort_values("predicted_change", key=abs, ascending=False)
    print(extreme_df.head(10).to_string(index=False))

print("\n\n==============================")
print("Verdict")
print("==============================")

if len(flags["crashed"]) == 0 and len(flags["impossible_outcome"]) == 0:
    print(
        "\nNo crashes and no physically impossible outcome values "
        "across the full sample -- the guardrails already in place "
        "are holding up under broad, unsupervised real-world use, "
        "not just the recurring demo cases."
    )
else:
    print(
        f"\n{len(flags['crashed'])} crashes and {len(flags['impossible_outcome'])} "
        f"impossible-outcome cases found -- worth investigating each "
        f"one specifically before considering this ready to package."
    )
