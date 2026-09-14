"""
39_validate_pitch_type_sensitivity.py

Purpose:
--------
38_similarity_weighted_regression.py's single demonstration showed
something worth checking more broadly: predicted WAR effect for
"add a slider" (+0.501) and "add a sweeper" (+0.496) came out nearly
identical for the same target pitcher. Three possible explanations,
not mutually exclusive:
1. A real finding -- for pitchers with this profile, adding almost
   any complementary breaking ball has a similar effect
2. A shared-comp artifact -- the same handful of similar pitchers
   (who made BOTH kinds of changes across their careers) drive both
   predictions, so some convergence isn't surprising
3. A real limitation of observational data -- this can't fully
   separate "the pitch helped" from "pitchers who add pitches were
   often already trending up for other reasons"

This runs the SAME prediction across several DIVERSE target pitchers
(stratified by max_velo_z, for genuine profile diversity, not just
whichever pitchers happened to load first) and several pitch types
with enough historical precedent to support a real prediction, then
checks:
1. WITHIN-target variance: for a given target pitcher, how much does
   the predicted effect vary across different pitch types? If this
   stays small for EVERY target tested, that's evidence the pattern
   from the single demonstration is systematic, not a one-off.
2. BETWEEN-target variance: for a given pitch type, how much does
   the predicted effect vary across different target pitchers? If
   this is much LARGER than within-target variance, the model may be
   picking up mostly on "which pitcher/which comps" rather than
   "which specific pitch."
3. COMP OVERLAP: for a given target, how much do the top comps
   overlap across different pitch-type queries? High overlap
   directly supports explanation #2 above.

Imports training data and the prediction function directly from
38_similarity_weighted_regression.py (via importlib, since Python
module names can't start with a digit) rather than duplicating that
logic.

Output:
-------
Printed diagnostic tables only -- no file saved. This is meant to
inform whether the tool needs further refinement before being
trusted for real recommendations, not to produce a deliverable
itself.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 38 (numeric-prefixed filename needs importlib, not
# a plain `import` statement)
# ============================================================

print("Loading training data and prediction function from 38...")

spec = importlib.util.spec_from_file_location(
    "similarity_weighted_regression", "38_similarity_weighted_regression.py"
)
swr = importlib.util.module_from_spec(spec)
sys.modules["similarity_weighted_regression"] = swr
spec.loader.exec_module(swr)  # runs 38's module-level code (data
                                # loading + function defs), but NOT
                                # its demonstration block, which is
                                # guarded by if __name__=="__main__"

training = swr.training
similarity_features = swr.similarity_features
outcomes = swr.outcomes
predict_arsenal_change_effect = swr.predict_arsenal_change_effect
SIMILARITY_FEATURE_COLS = swr.SIMILARITY_FEATURE_COLS
MIN_GROUP_SIZE_FOR_REGRESSION = swr.MIN_GROUP_SIZE_FOR_REGRESSION


# ============================================================
# SETTINGS
# ============================================================

N_TARGET_PITCHERS = 6
CHANGE_TYPE = "ADD"  # kept fixed across the whole validation -- only
                       # varying pitch_type and target pitcher, so any
                       # variation observed is attributable to those,
                       # not a third moving part

# DEFAULT METRIC PREFERENCE, based on real evidence from
# 41_outcome_metric_comparison.py: xwoba_against showed the highest
# sign consistency (75.0% vs. WAR's 56.3%) -- WAR carries a whole
# season's worth of unrelated context (defense, luck, innings
# distribution), diluting the change-specific signal relative to
# metrics closer to what a pitch itself does. Matches
# 38_similarity_weighted_regression.py's updated default ordering.
OUTCOME_METRIC = next(
    (c for c in ["xwoba_against", "fip", "war"] if f"{c}_before" in training.columns),
    None
)

if OUTCOME_METRIC is None:
    raise ValueError("No outcome metric available in the training data.")

print(f"Using outcome metric: {OUTCOME_METRIC}")


# ============================================================
# SELECT DIVERSE TARGET PITCHERS
# ============================================================

print(f"\nSelecting {N_TARGET_PITCHERS} diverse target pitchers (stratified by max_velo_z)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
complete_profiles = complete_profiles.sort_values("max_velo_z").reset_index(drop=True)

# evenly spaced across the max_velo_z range -- genuine profile
# diversity along a clear, interpretable dimension, not just
# whichever rows happen to load first
strat_indices = np.linspace(0, len(complete_profiles) - 1, N_TARGET_PITCHERS).astype(int)
target_rows = complete_profiles.iloc[strat_indices]

print(f"Selected targets (player, season, max_velo_z):")
for _, r in target_rows.iterrows():
    print(f"  {r['player_name']}, {r['season']} (max_velo_z={r['max_velo_z']:.2f})")


# ============================================================
# DETERMINE WHICH PITCH TYPES HAVE ENOUGH PRECEDENT
# ============================================================

add_counts = training[training["change_type"] == CHANGE_TYPE]["pitch_type"].value_counts()
viable_pitch_types = add_counts[add_counts >= MIN_GROUP_SIZE_FOR_REGRESSION].index.tolist()

print(f"\nPitch types with enough '{CHANGE_TYPE}' precedent for a real regression: {viable_pitch_types}")

if len(viable_pitch_types) < 2:
    raise ValueError(
        "Fewer than 2 pitch types have enough precedent to run a "
        "meaningful sensitivity check."
    )


# ============================================================
# RUN THE FULL GRID
# ============================================================

print(f"\nRunning {len(target_rows)} targets x {len(viable_pitch_types)} pitch types...")

results = []
comps_by_target_pitchtype = {}

for _, target_row in target_rows.iterrows():
    target_features = {c: target_row[c] for c in SIMILARITY_FEATURE_COLS}
    target_features["usage_pct_delta"] = 10.0

    target_baseline_rows = outcomes[outcomes["player_id"] == target_row["player_id"]]
    target_outcome_before = (
        target_baseline_rows[OUTCOME_METRIC].iloc[0]
        if len(target_baseline_rows) > 0 and OUTCOME_METRIC in target_baseline_rows.columns
        else training[f"{OUTCOME_METRIC}_before"].median()
    )

    for pitch_type in viable_pitch_types:
        result = predict_arsenal_change_effect(
            target_features, pitch_type, CHANGE_TYPE,
            target_outcome_before, OUTCOME_METRIC,
            target_row["player_id"], target_row["season"]
        )

        top_comp_names = set()
        if result["predicted_outcome"] is not None and "combined_weight" in result["group"].columns:
            top_comp_names = set(
                result["group"].nlargest(5, "combined_weight")["player_name"].tolist()
            )

        target_key = f"{target_row['player_name']}_{target_row['season']}"
        comps_by_target_pitchtype[(target_key, pitch_type)] = top_comp_names

        results.append({
            "target": target_key,
            "target_max_velo_z": target_row["max_velo_z"],
            "pitch_type": pitch_type,
            "predicted_change": result["predicted_change"] if result["predicted_outcome"] is not None else np.nan,
            "n_historical_events": result["n_historical_events"],
            "method": result["method"],
            "mean_reversion_coef": result.get("mean_reversion_coef"),
            "change_specific_coef": result.get("change_specific_coef"),
            "new_pitch_distance_coef": result.get("new_pitch_distance_coef"),
            "synergy_coef": result.get("synergy_coef"),
            "used_distance_feature": result.get("used_distance_feature", False),
            "used_synergy_feature": result.get("used_synergy_feature", False)
        })

results_df = pd.DataFrame(results)


# ============================================================
# CHECK 1: WITHIN-TARGET VARIANCE (across pitch types, same pitcher)
# ============================================================

print("\n\n==============================")
print("CHECK 1: Within-target variance (across pitch types, same pitcher)")
print("==============================")
print(
    "If predicted_change_std stays small for EVERY target, that's "
    "evidence the pattern from the single demonstration is "
    "systematic, not a one-off.\n"
)

within_target = results_df.groupby("target")["predicted_change"].agg(["mean", "std", "min", "max"])
within_target["range"] = within_target["max"] - within_target["min"]
print(within_target.round(4))


# ============================================================
# CHECK 2: BETWEEN-TARGET VARIANCE (across pitchers, same pitch type)
# ============================================================

print("\n\n==============================")
print("CHECK 2: Between-target variance (across pitchers, same pitch type)")
print("==============================")
print(
    "Compare this to Check 1's std values -- if between-target "
    "variance is much LARGER, the model may be picking up mostly "
    "on 'which pitcher' rather than 'which pitch.'\n"
)

between_target = results_df.groupby("pitch_type")["predicted_change"].agg(["mean", "std", "min", "max"])
between_target["range"] = between_target["max"] - between_target["min"]
print(between_target.round(4))

avg_within_std = within_target["std"].mean()
avg_between_std = between_target["std"].mean()
print(
    f"\nAverage within-target std: {avg_within_std:.4f}"
    f"\nAverage between-target std: {avg_between_std:.4f}"
)
if avg_between_std > avg_within_std * 1.5:
    print(
        "Between-target variance is notably larger than within-"
        "target variance -- consistent with the model responding "
        "more to WHICH PITCHER than WHICH PITCH."
    )
elif avg_within_std > avg_between_std * 1.5:
    print(
        "Within-target variance is notably larger than between-"
        "target variance -- the model DOES meaningfully "
        "differentiate between pitch types for a given pitcher, "
        "contrary to what the single demonstration suggested."
    )
else:
    print("Within- and between-target variance are of a similar order.")


# ============================================================
# CHECK 3: COMP OVERLAP ACROSS PITCH-TYPE QUERIES (same target)
# ============================================================

print("\n\n==============================")
print("CHECK 3: Top-comp overlap across pitch types (same target)")
print("==============================")
print(
    "High overlap directly supports the 'shared-comp artifact' "
    "explanation -- the same historical pitchers driving predictions "
    "regardless of which specific pitch is being queried.\n"
)

for target_key in results_df["target"].unique():
    pitch_types_for_target = [pt for (tk, pt) in comps_by_target_pitchtype if tk == target_key]
    if len(pitch_types_for_target) < 2:
        continue

    print(f"\n{target_key}:")
    for i in range(len(pitch_types_for_target)):
        for j in range(i + 1, len(pitch_types_for_target)):
            pt_a, pt_b = pitch_types_for_target[i], pitch_types_for_target[j]
            comps_a = comps_by_target_pitchtype[(target_key, pt_a)]
            comps_b = comps_by_target_pitchtype[(target_key, pt_b)]
            if not comps_a or not comps_b:
                continue
            overlap = comps_a & comps_b
            print(
                f"  {pt_a} vs {pt_b}: {len(overlap)} of "
                f"{min(len(comps_a), len(comps_b))} top comps shared "
                f"{sorted(overlap) if overlap else ''}"
            )


# ============================================================
# CHECK 4: MEAN-REVERSION VS. CHANGE-SPECIFIC COEFFICIENT STABILITY
# ============================================================

print("\n\n==============================")
print("CHECK 4: Mean-reversion vs. change-specific coefficient stability")
print("==============================")
print(
    "Both coefficients are on STANDARDIZED predictors (see\n"
    "38's updated regression step), so they're directly comparable\n"
    "in magnitude, not distorted by outcome-vs-usage-delta being on\n"
    "different raw scales. Tests the hypothesis directly, rather\n"
    "than inferring it indirectly from the final blended prediction:\n"
    "if mean_reversion_coef stays large and consistent across pitch\n"
    "types for a given target, while change_specific_coef is small\n"
    "and/or sign-inconsistent, that confirms the regression is\n"
    "mostly picking up mean-reversion, not a genuine pitch-specific\n"
    "effect.\n"
)

coef_rows = results_df[results_df["mean_reversion_coef"].notna()].copy()

if len(coef_rows) == 0:
    print("No weighted_regression fits available to analyze (all fell back to the weighted-average method).")
else:
    coef_summary = coef_rows.groupby("target").agg(
        mean_reversion_mean=("mean_reversion_coef", "mean"),
        mean_reversion_std=("mean_reversion_coef", "std"),
        change_specific_mean=("change_specific_coef", "mean"),
        change_specific_std=("change_specific_coef", "std"),
    )
    # sign consistency: what fraction of this target's change_specific_coef
    # values share the same sign as the majority -- a coefficient that's
    # genuinely picking up a real effect should mostly agree in sign
    # across pitch types; one that's just noise should flip more freely
    def sign_consistency(vals):
        # REAL BUG FIX: the previous version used np.sign(vals.mean())
        # as the "majority" reference -- but a single large-magnitude
        # outlier can pull the MEAN to one sign even when individual
        # VALUES are numerically dominated by the other sign. Confirmed
        # on real data: Ben Rowen's change_specific_coef had 5 positive
        # values and 3 negative, but one large negative outlier
        # (FS=-0.2719) pulled the mean negative, causing the old
        # version to report 3/8 "consistency" (agreement with the
        # NEGATIVE minority) instead of the correct 5/8 (agreement
        # with the TRUE positive majority). This directly compares
        # each value's sign to a majority determined by counting
        # signs, not by the mean's sign -- guaranteed >=50% by
        # construction, matching genuine majority consensus.
        vals = vals.dropna()
        if len(vals) == 0:
            return np.nan
        signs = np.sign(vals)
        n_positive = (signs > 0).sum()
        n_negative = (signs < 0).sum()
        majority_sign = 1 if n_positive >= n_negative else -1
        return (signs == majority_sign).mean()

    coef_summary["change_specific_sign_consistency"] = coef_rows.groupby("target")["change_specific_coef"].apply(sign_consistency)

    print(coef_summary.round(4))

    avg_mean_reversion_mag = coef_rows["mean_reversion_coef"].abs().mean()
    avg_change_specific_mag = coef_rows["change_specific_coef"].abs().mean()
    avg_sign_consistency = coef_summary["change_specific_sign_consistency"].mean()

    print(
        f"\nAverage |mean_reversion_coef|: {avg_mean_reversion_mag:.4f}"
        f"\nAverage |change_specific_coef|: {avg_change_specific_mag:.4f}"
        f"\nAverage change_specific sign consistency across targets: {avg_sign_consistency:.1%}"
    )

    if avg_mean_reversion_mag > avg_change_specific_mag * 2:
        print(
            "\nmean_reversion_coef is more than 2x the magnitude of "
            "change_specific_coef on average -- consistent with the "
            "hypothesis that the regression is dominated by mean-"
            "reversion rather than a genuine, strong pitch-specific "
            "effect."
        )
    if avg_sign_consistency < 0.7:
        print(
            "change_specific_coef's sign is inconsistent across pitch "
            "types for the average target (agrees with the majority "
            "sign less than 70% of the time) -- consistent with this "
            "coefficient being mostly noise rather than a stable, "
            "trustworthy effect."
        )

    # ============================================================
    # CHECK 4b: DOES THE NEW new_pitch_distance_coef HELP?
    # ============================================================
    print("\n\n==============================")
    print("CHECK 4b: Does new_pitch_nn_distance_from_existing carry real signal?")
    print("==============================")
    print(
        "Only populated for rows using 'weighted_regression_with_distance'\n"
        "(40_new_pitch_distance_features.py's feature, ADD events with\n"
        "enough coverage). Same sign-consistency and magnitude checks\n"
        "as change_specific_coef above -- the direct test of whether\n"
        "this new feature actually gives the change-specific term more\n"
        "to work with, rather than just assuming it helped.\n"
    )

    distance_rows = results_df[results_df["used_distance_feature"] == True].copy()

    if len(distance_rows) == 0:
        print(
            "No rows used the distance-feature regression this run -- "
            "either 40_new_pitch_distance_features.py hasn't been run, "
            "or no (target, pitch_type) combination had enough ADD "
            "events with the distance feature populated to clear "
            "MIN_GROUP_SIZE_FOR_REGRESSION."
        )
    else:
        distance_summary = distance_rows.groupby("target").agg(
            new_pitch_distance_mean=("new_pitch_distance_coef", "mean"),
            new_pitch_distance_std=("new_pitch_distance_coef", "std"),
        )
        distance_summary["new_pitch_distance_sign_consistency"] = (
            distance_rows.groupby("target")["new_pitch_distance_coef"].apply(sign_consistency)
        )
        print(distance_summary.round(4))

        avg_distance_mag = distance_rows["new_pitch_distance_coef"].abs().mean()
        avg_distance_sign_consistency = distance_summary["new_pitch_distance_sign_consistency"].mean()

        # compare directly against change_specific_coef WITHIN THE SAME
        # subset of rows (only those that used the distance feature),
        # not the broader coef_rows average -- an apples-to-apples
        # comparison of "with vs. without" on the identical row set
        avg_change_specific_mag_same_rows = distance_rows["change_specific_coef"].abs().mean()

        print(
            f"\nAverage |new_pitch_distance_coef|: {avg_distance_mag:.4f}"
            f"\nAverage |change_specific_coef| (same rows, for comparison): {avg_change_specific_mag_same_rows:.4f}"
            f"\nAverage new_pitch_distance sign consistency: {avg_distance_sign_consistency:.1%}"
        )

        if avg_distance_mag > avg_change_specific_mag_same_rows and avg_distance_sign_consistency > avg_sign_consistency:
            print(
                "\nnew_pitch_distance_coef is BOTH larger in magnitude AND "
                "more sign-consistent than change_specific_coef on the "
                "same rows -- real evidence this feature gives the model "
                "more genuine pitch-specific signal to work with."
            )
        elif avg_distance_sign_consistency <= avg_sign_consistency:
            print(
                "\nnew_pitch_distance_coef is NOT more sign-consistent "
                "than change_specific_coef was -- this feature may not "
                "be solving the problem on its own; worth investigating "
                "further rather than assuming it fixed things."
            )

    # ============================================================
    # CHECK 4c: DOES THE NEW arsenal_synergy_whiff_delta HELP?
    # ============================================================
    print("\n\n==============================")
    print("CHECK 4c: Does arsenal_synergy_whiff_delta carry real signal?")
    print("==============================")
    print(
        "Tests the tunneling/setup-value hypothesis directly: does\n"
        "measuring whether the pitcher's OTHER pitches improved or\n"
        "declined after the new pitch was added (42_arsenal_synergy_\n"
        "features.py) give the model more genuine pitch-specific\n"
        "signal than usage_pct_delta alone? Same sign-consistency and\n"
        "magnitude checks as Check 4b.\n"
    )

    synergy_rows = results_df[results_df["used_synergy_feature"] == True].copy()

    if len(synergy_rows) == 0:
        print(
            "No rows used the synergy-feature regression this run -- "
            "either 42_arsenal_synergy_features.py hasn't been run, "
            "or no (target, pitch_type) combination had enough ADD "
            "events with the synergy feature populated to clear "
            "MIN_GROUP_SIZE_FOR_REGRESSION."
        )
    else:
        synergy_summary = synergy_rows.groupby("target").agg(
            synergy_coef_mean=("synergy_coef", "mean"),
            synergy_coef_std=("synergy_coef", "std"),
        )
        synergy_summary["synergy_sign_consistency"] = (
            synergy_rows.groupby("target")["synergy_coef"].apply(sign_consistency)
        )
        print(synergy_summary.round(4))

        avg_synergy_mag = synergy_rows["synergy_coef"].abs().mean()
        avg_synergy_sign_consistency = synergy_summary["synergy_sign_consistency"].mean()

        # same-rows comparison against change_specific_coef, matching
        # Check 4b's apples-to-apples approach
        avg_change_specific_mag_same_rows_synergy = synergy_rows["change_specific_coef"].abs().mean()

        print(
            f"\nAverage |synergy_coef|: {avg_synergy_mag:.4f}"
            f"\nAverage |change_specific_coef| (same rows, for comparison): {avg_change_specific_mag_same_rows_synergy:.4f}"
            f"\nAverage synergy sign consistency: {avg_synergy_sign_consistency:.1%}"
        )

        if avg_synergy_mag > avg_change_specific_mag_same_rows_synergy and avg_synergy_sign_consistency > avg_sign_consistency:
            print(
                "\nsynergy_coef is BOTH larger in magnitude AND more "
                "sign-consistent than change_specific_coef on the same "
                "rows -- real evidence the tunneling/setup-value "
                "mechanism gives the model more genuine pitch-specific "
                "signal to work with."
            )
        elif avg_synergy_sign_consistency <= avg_sign_consistency:
            print(
                "\nsynergy_coef is NOT more sign-consistent than "
                "change_specific_coef was -- this feature may not be "
                "solving the problem on its own; worth investigating "
                "further rather than assuming it fixed things."
            )


print("\n\n==============================")
print("Full results table")
print("==============================")
print(results_df.round(4).to_string(index=False))
