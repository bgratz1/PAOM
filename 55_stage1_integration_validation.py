"""
55_stage1_integration_validation.py

Purpose:
--------
54_stage2_integration.py's single demonstration showed
stage1_quality_coef EXCEEDING mean_reversion_coef for 2 of 3
candidates -- if real, this would be the strongest change-specific
signal this whole thread has produced. But three data points on one
target pitcher is exactly the situation that's misled this project
before (the very first single-demonstration result that kicked off
the whole mean-reversion investigation). This applies the same
rigor already used for every other feature: a Check-4-style test
across multiple diverse targets, PLUS a placebo test (same
permutation-test technique as 43/44) confirming the result isn't an
artifact.

A REAL TECHNICAL SUBTLETY, handled carefully rather than assumed:
pitch_level_indexed is referenced as a free variable inside BOTH
50's predict_new_pitch_quality (Stage 1's own internal lookups) AND
54's predict_arsenal_change_with_stage1 (the historical group's
ground-truth lookups) -- each resolved against ITS OWN DEFINING
MODULE's namespace, not a shared object. A placebo shuffle has to
monkey-patch BOTH module-level references to the SAME shuffled
series, or the test would be silently incomplete (only scrambling
half of what the integration actually uses). Explicitly verified
below before trusting the comparison.

Output:
-------
Printed magnitude and sign-consistency comparison for stage1_
quality_coef vs. mean_reversion_coef and change_specific_coef,
across multiple targets, plus the placebo (shuffled) comparison. No
file saved -- this is a validation check, not a deliverable.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 54 (which cascades through 50, 45, 38)
# ============================================================

print("Loading Stage 1 + Stage 2 integration from 54...")

spec = importlib.util.spec_from_file_location(
    "stage2_integration", "54_stage2_integration.py"
)
stage2 = importlib.util.module_from_spec(spec)
sys.modules["stage2_integration"] = stage2
spec.loader.exec_module(stage2)

training = stage2.training
similarity_features = stage2.similarity_features
outcomes = stage2.outcomes
predict_arsenal_change_with_stage1 = stage2.predict_arsenal_change_with_stage1
estimate_realistic_usage_delta = stage2.estimate_realistic_usage_delta
SIMILARITY_FEATURE_COLS = stage2.SIMILARITY_FEATURE_COLS

# the SAME underlying stage1 module 54 imported -- needed to patch
# ITS copy of pitch_level_indexed too (see docstring above)
stage1 = stage2.stage1


# ============================================================
# SETTINGS
# ============================================================

N_TARGET_PITCHERS = 6  # same general philosophy as 39/41/52
VIABLE_PITCH_TYPES = ["SI", "SL", "ST", "FS", "FC", "CH", "CU", "FF"]
OUTCOME_METRIC = "xwoba_against"

RANDOM_SEED = 42


# ============================================================
# SAME SIGN-CONSISTENCY LOGIC AS 39 (corrected, count-based majority)
# ============================================================

def sign_consistency(vals):
    vals = pd.Series(vals).dropna()
    if len(vals) == 0:
        return np.nan
    signs = np.sign(vals)
    n_positive = (signs > 0).sum()
    n_negative = (signs < 0).sum()
    majority_sign = 1 if n_positive >= n_negative else -1
    return (signs == majority_sign).mean()


# ============================================================
# SELECT DIVERSE TARGET PITCHERS
# ============================================================

print(f"\nSelecting {N_TARGET_PITCHERS} diverse target pitchers (stratified by max_velo_z)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
complete_profiles = complete_profiles.sort_values("max_velo_z").reset_index(drop=True)

strat_indices = np.linspace(0, len(complete_profiles) - 1, N_TARGET_PITCHERS).astype(int)
target_rows = complete_profiles.iloc[strat_indices]

for _, r in target_rows.iterrows():
    print(f"  {r['player_name']}, {r['season']} (max_velo_z={r['max_velo_z']:.2f})")


# ============================================================
# RUN THE GRID (used for both real and placebo passes)
# ============================================================

def run_grid(label):
    print(f"\nRunning grid ({label})...")
    rows = []
    for _, target_row in target_rows.iterrows():
        target_key = f"{target_row['player_name']}_{target_row['season']}"
        target_features = {c: target_row[c] for c in SIMILARITY_FEATURE_COLS}

        target_baseline_rows = outcomes[outcomes["player_id"] == target_row["player_id"]]
        if len(target_baseline_rows) == 0 or OUTCOME_METRIC not in target_baseline_rows.columns:
            continue
        target_outcome_before = target_baseline_rows.sort_values("season")[OUTCOME_METRIC].iloc[-1]

        for pitch_type in VIABLE_PITCH_TYPES:
            realistic_delta, _ = estimate_realistic_usage_delta(
                target_features, pitch_type, "ADD", target_row["player_id"],
                target_row["season"], target_outcome_before, OUTCOME_METRIC
            )
            if realistic_delta is None:
                continue

            final_features = dict(target_features)
            final_features["usage_pct_delta"] = realistic_delta

            result = predict_arsenal_change_with_stage1(
                final_features, pitch_type, "ADD", target_outcome_before,
                OUTCOME_METRIC, target_row["player_id"], target_row["season"]
            )

            if result.get("stage1_integration_status") == "success":
                rows.append({
                    "target": target_key,
                    "pitch_type": pitch_type,
                    "mean_reversion_coef": result.get("mean_reversion_coef"),
                    "change_specific_coef": result.get("change_specific_coef"),
                    "stage1_quality_coef": result.get("stage1_quality_coef"),
                })

    return pd.DataFrame(rows)


real_results = run_grid("REAL")
print(f"{len(real_results):,} successful integrated predictions")


# ============================================================
# REAL MAGNITUDE + SIGN CONSISTENCY
# ============================================================

print("\n\n==============================")
print("Real results: stage1_quality_coef vs. mean_reversion_coef vs. change_specific_coef")
print("==============================")

if len(real_results) < 5:
    raise ValueError(f"Only {len(real_results)} successful predictions -- too few to validate meaningfully.")

avg_mean_reversion = real_results["mean_reversion_coef"].abs().mean()
avg_change_specific = real_results["change_specific_coef"].abs().mean()
avg_stage1 = real_results["stage1_quality_coef"].abs().mean()

stage1_sign_consistency = real_results.groupby("target")["stage1_quality_coef"].apply(sign_consistency).mean()

print(f"\nAverage |mean_reversion_coef|:  {avg_mean_reversion:.4f}")
print(f"Average |change_specific_coef|: {avg_change_specific:.4f}")
print(f"Average |stage1_quality_coef|:  {avg_stage1:.4f}")
print(f"\nstage1_quality_coef sign consistency: {stage1_sign_consistency:.1%}")

print(f"\nFull per-target-pitch-type breakdown:")
print(real_results.round(4).to_string(index=False))


# ============================================================
# PLACEBO TEST -- shuffle the REAL ground-truth pitch quality
# used by BOTH 50's internal lookups and 54's own lookups
# ============================================================

print("\n\n==============================")
print("Placebo check (shuffled ground-truth pitch quality)")
print("==============================")

rng = np.random.default_rng(RANDOM_SEED)

real_pitch_level_indexed = stage1.pitch_level_indexed
shuffled_values = rng.permutation(real_pitch_level_indexed.values)
shuffled_pitch_level_indexed = pd.Series(shuffled_values, index=real_pitch_level_indexed.index)

# BOTH module-level references must be patched -- confirmed by
# tracing which module's globals each function resolves
# pitch_level_indexed against (see docstring)
stage1.pitch_level_indexed = shuffled_pitch_level_indexed
stage2.pitch_level_indexed = shuffled_pitch_level_indexed

# sanity check the patch actually took effect before trusting the
# placebo run built on top of it
assert stage1.pitch_level_indexed.iloc[0] == shuffled_pitch_level_indexed.iloc[0]
assert stage2.pitch_level_indexed.iloc[0] == shuffled_pitch_level_indexed.iloc[0]
assert not stage1.pitch_level_indexed.equals(real_pitch_level_indexed)
print("Confirmed: both module references successfully patched to the shuffled series")

placebo_results = run_grid("PLACEBO")

# restore real data afterward, in case this module stays loaded
stage1.pitch_level_indexed = real_pitch_level_indexed
stage2.pitch_level_indexed = real_pitch_level_indexed

print(f"\n{len(placebo_results):,} successful placebo predictions")

if len(placebo_results) >= 5:
    placebo_avg_stage1 = placebo_results["stage1_quality_coef"].abs().mean()
    placebo_sign_consistency = placebo_results.groupby("target")["stage1_quality_coef"].apply(sign_consistency).mean()

    print(f"\nReal |stage1_quality_coef|:    {avg_stage1:.4f}")
    print(f"Placebo |stage1_quality_coef|: {placebo_avg_stage1:.4f}")
    print(f"\nReal sign consistency:    {stage1_sign_consistency:.1%}")
    print(f"Placebo sign consistency: {placebo_sign_consistency:.1%}")

    if avg_stage1 > placebo_avg_stage1 * 1.3 and stage1_sign_consistency > placebo_sign_consistency + 0.15:
        print(
            "\nReal result meaningfully exceeds the placebo on BOTH "
            "magnitude and sign consistency -- genuine evidence "
            "stage1_quality_coef carries real signal, not an artifact."
        )
    else:
        print(
            "\nReal result does NOT clearly exceed the placebo -- "
            "the single-demonstration result may have been "
            "misleading, the same trap this project has hit before. "
            "Worth treating stage1_quality_coef's apparent strength "
            "with real skepticism until this resolves more clearly."
        )
else:
    print("Too few placebo predictions to compare meaningfully.")
