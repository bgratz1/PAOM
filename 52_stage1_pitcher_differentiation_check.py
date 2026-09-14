"""
52_stage1_pitcher_differentiation_check.py

Purpose:
--------
51_stage1_validation.py's real output hinted at something worth
confirming formally rather than eyeballing: for a given pitch type,
predicted_pitch_quality barely varied across 8 completely different
real pitchers (e.g. SI: 0.351-0.353, a span of 0.002), while their
ACTUAL outcomes for that same pitch spanned 0.231-0.423 -- nearly
100x wider. This directly tests the question that matters: does
Stage 1 predict a DIFFERENT slider for Pitcher A vs. Pitcher B, or
does it mostly just predict "sliders are generally around X,
whoever's asking"?

Same within-vs-between variance decomposition already used in
39_validate_pitch_type_sensitivity.py's Check 1/Check 2, applied
here with the roles reversed: for STAGE 1 specifically, real
BETWEEN-pitcher variance (for the SAME pitch type) is what we WANT
to see -- that's literally what "does this pitch look different for
different pitchers" means. If within-target (same pitcher, across
pitch types) variance dominates instead, that's evidence Stage 1 is
mostly capturing "which pitch type" rather than "which pitcher."

Output:
-------
Printed within-target vs. between-target variance comparison, plus
an explicit verdict. No file saved -- this is a diagnostic, not a
deliverable.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 50
# ============================================================

print("Loading training data and Stage 1 prediction function from 50...")

spec = importlib.util.spec_from_file_location(
    "stage1_pitch_quality_model", "50_stage1_pitch_quality_model.py"
)
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1_pitch_quality_model"] = stage1
spec.loader.exec_module(stage1)

training = stage1.training
similarity_features = stage1.similarity_features
predict_new_pitch_quality = stage1.predict_new_pitch_quality
SIMILARITY_FEATURE_COLS = stage1.SIMILARITY_FEATURE_COLS


# ============================================================
# SETTINGS
# ============================================================

N_TARGET_PITCHERS = 8  # same general philosophy as 39/41 -- stratified
                         # for genuine profile diversity, not whichever
                         # rows happen to load first

VIABLE_PITCH_TYPES = ["SI", "SL", "ST", "FS", "FC", "CH", "CU", "FF"]


# ============================================================
# SELECT DIVERSE TARGET PITCHERS (same stratified approach as 39/41)
# ============================================================

print(f"\nSelecting {N_TARGET_PITCHERS} diverse target pitchers (stratified by max_velo_z)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
complete_profiles = complete_profiles.sort_values("max_velo_z").reset_index(drop=True)

strat_indices = np.linspace(0, len(complete_profiles) - 1, N_TARGET_PITCHERS).astype(int)
target_rows = complete_profiles.iloc[strat_indices]

for _, r in target_rows.iterrows():
    print(f"  {r['player_name']}, {r['season']} (max_velo_z={r['max_velo_z']:.2f})")


# ============================================================
# RUN STAGE 1 FOR EVERY (TARGET, PITCH TYPE) COMBINATION
# ============================================================

print(f"\nRunning Stage 1 for {N_TARGET_PITCHERS} targets x {len(VIABLE_PITCH_TYPES)} pitch types...")

results = []

for _, target_row in target_rows.iterrows():
    target_key = f"{target_row['player_name']}_{target_row['season']}"
    for pitch_type in VIABLE_PITCH_TYPES:
        result = predict_new_pitch_quality(
            target_row["player_id"], target_row["season"], pitch_type
        )
        if result["predicted_pitch_quality"] is not None:
            results.append({
                "target": target_key,
                "pitch_type": pitch_type,
                "predicted_pitch_quality": result["predicted_pitch_quality"],
                "n_comps": result.get("n_comps_with_pitch_level_data", np.nan),
            })

results_df = pd.DataFrame(results)

print(f"\n{len(results_df):,} usable predictions produced")


# ============================================================
# CHECK 1: WITHIN-TARGET VARIANCE (across pitch types, same pitcher)
# ============================================================

print("\n\n==============================")
print("CHECK 1: Within-target variance (across pitch types, same pitcher)")
print("==============================")
print(
    "How much does Stage 1's prediction vary for ONE pitcher across\n"
    "different pitch types they could add.\n"
)

within_target = results_df.groupby("target")["predicted_pitch_quality"].agg(["mean", "std", "min", "max"])
within_target["range"] = within_target["max"] - within_target["min"]
print(within_target.round(4))


# ============================================================
# CHECK 2: BETWEEN-TARGET VARIANCE (across pitchers, same pitch type)
# ============================================================

print("\n\n==============================")
print("CHECK 2: Between-target variance (across pitchers, same pitch type)")
print("==============================")
print(
    "THE key question: does Stage 1 predict a DIFFERENT slider for\n"
    "different pitchers, or mostly just 'sliders are generally around\n"
    "X'? This is what real BETWEEN-pitcher variance for the SAME\n"
    "pitch type actually measures.\n"
)

between_target = results_df.groupby("pitch_type")["predicted_pitch_quality"].agg(["mean", "std", "min", "max"])
between_target["range"] = between_target["max"] - between_target["min"]
print(between_target.round(4))

avg_within_std = within_target["std"].mean()
avg_between_std = between_target["std"].mean()

print(
    f"\nAverage within-target std (differentiation BY PITCH TYPE): {avg_within_std:.4f}"
    f"\nAverage between-target std (differentiation BY PITCHER): {avg_between_std:.4f}"
)


# ============================================================
# VERDICT
# ============================================================

print("\n\n==============================")
print("Verdict")
print("==============================")

if avg_between_std < avg_within_std * 0.3:
    print(
        f"\nBetween-target (by-pitcher) variance is dramatically SMALLER "
        f"than within-target (by-pitch-type) variance -- CONFIRMS the "
        f"pattern seen in 51's real output: Stage 1 differentiates "
        f"MUCH more by which pitch type is being added than by which "
        f"specific pitcher is asking. Two different pitchers adding "
        f"the same pitch type get nearly the same prediction, even "
        f"though their real outcomes for that pitch type vary widely."
    )
elif avg_between_std < avg_within_std:
    print(
        f"\nBetween-target (by-pitcher) variance is smaller than "
        f"within-target (by-pitch-type) variance, though not as "
        f"dramatically as feared -- Stage 1 differentiates SOME by "
        f"pitcher, but pitch-type identity is still the dominant "
        f"factor in its predictions."
    )
else:
    print(
        f"\nBetween-target (by-pitcher) variance is comparable to or "
        f"larger than within-target (by-pitch-type) variance -- "
        f"Stage 1 DOES meaningfully differentiate between different "
        f"pitchers adding the same pitch, contrary to what the "
        f"single validation run's sample suggested."
    )

print(f"\n\nFull results table:")
print(results_df.round(4).to_string(index=False))
