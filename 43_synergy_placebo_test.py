"""
43_synergy_placebo_test.py

Purpose:
--------
Check 4c (39_validate_pitch_type_sensitivity.py) found arsenal_
synergy_whiff_delta with 100.0% sign consistency across 48 real
predictions -- the strongest result this whole thread has produced.
But there's a real, specific reason to be suspicious of it that
doesn't apply to the earlier distance feature: arsenal_synergy_
whiff_delta is computed from the pitcher's OTHER pitches' whiff_rate
in season_to -- the SAME season as the outcome (xwoba_against in
season_to) being predicted. A pitcher having a genuinely great
season shows up simultaneously as higher whiff rates across their
OTHER pitches AND lower overall xwOBA-against -- not necessarily
because the new pitch caused tunneling-driven improvement in the
other pitches, but because both are just two different facets of
"this was a good season for this pitcher." The distance feature
never had this problem -- it's computed entirely from movement
profiles that exist independently of how the season turned out.

THIS TEST: a standard permutation-test / label-shuffling control.
Shuffle the REAL arsenal_synergy_whiff_delta values across ADD
events -- this breaks the true pairing between a specific pitcher's
actual synergy score and their actual outcome, while preserving the
EXACT SAME distribution of values (same mean, same spread, same
range). If this causally-meaningless SHUFFLED version still produces
strong magnitude and high sign-consistency, that's direct evidence
the original result was an artifact of scale, distribution, or
regression mechanics -- NOT genuine pitcher-specific signal. If the
shuffled version collapses toward weak/inconsistent, that validates
the real feature's strength as genuine.

Implementation: predict_arsenal_change_effect() already accepts a
training_df override -- this builds a placebo copy of the training
data with synergy values shuffled, and runs the IDENTICAL target x
pitch_type grid against both the real and placebo versions for a
direct, apples-to-apples comparison.

Output:
-------
Side-by-side comparison of real vs. placebo synergy_coef magnitude
and sign consistency. No file saved -- this is a validation check,
not a deliverable.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 38
# ============================================================

print("Loading training data and prediction function from 38...")

spec = importlib.util.spec_from_file_location(
    "similarity_weighted_regression", "38_similarity_weighted_regression.py"
)
swr = importlib.util.module_from_spec(spec)
sys.modules["similarity_weighted_regression"] = swr
spec.loader.exec_module(swr)

training = swr.training
similarity_features = swr.similarity_features
outcomes = swr.outcomes
predict_arsenal_change_effect = swr.predict_arsenal_change_effect
SIMILARITY_FEATURE_COLS = swr.SIMILARITY_FEATURE_COLS
MIN_GROUP_SIZE_FOR_REGRESSION = swr.MIN_GROUP_SIZE_FOR_REGRESSION

if "arsenal_synergy_whiff_delta" not in training.columns or training["arsenal_synergy_whiff_delta"].notna().sum() == 0:
    raise ValueError(
        "arsenal_synergy_whiff_delta not populated in training data -- "
        "run 42_arsenal_synergy_features.py first."
    )


# ============================================================
# SETTINGS
# ============================================================

N_TARGET_PITCHERS = 6
CHANGE_TYPE = "ADD"
OUTCOME_METRIC = "xwoba_against"  # same metric Check 4c used

RANDOM_SEED = 42  # fixed for reproducibility


# ============================================================
# BUILD THE PLACEBO TRAINING SET
# ============================================================

print("\nBuilding placebo training set (synergy values shuffled)...")

training_placebo = training.copy()

rng = np.random.default_rng(RANDOM_SEED)

valid_mask = training_placebo["arsenal_synergy_whiff_delta"].notna()
real_values = training_placebo.loc[valid_mask, "arsenal_synergy_whiff_delta"].values.copy()

shuffled_values = rng.permutation(real_values)

training_placebo.loc[valid_mask, "arsenal_synergy_whiff_delta"] = shuffled_values

# sanity check: same distribution, different pairing
print(
    f"Real synergy -- mean: {real_values.mean():.5f}, std: {real_values.std():.5f}"
)
print(
    f"Placebo synergy -- mean: {shuffled_values.mean():.5f}, std: {shuffled_values.std():.5f} "
    f"(should match real exactly -- same values, different order)"
)
n_unchanged = (training_placebo.loc[valid_mask, "arsenal_synergy_whiff_delta"].values == real_values).sum()
print(
    f"{n_unchanged:,} of {valid_mask.sum():,} rows happen to keep their "
    f"original value after shuffling (expected by chance, not a bug)"
)


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
# SELECT DIVERSE TARGET PITCHERS (same approach as 39)
# ============================================================

print(f"\nSelecting {N_TARGET_PITCHERS} diverse target pitchers (stratified by max_velo_z)...")

complete_profiles = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).copy()
complete_profiles = complete_profiles.sort_values("max_velo_z").reset_index(drop=True)

strat_indices = np.linspace(0, len(complete_profiles) - 1, N_TARGET_PITCHERS).astype(int)
target_rows = complete_profiles.iloc[strat_indices]

for _, r in target_rows.iterrows():
    print(f"  {r['player_name']}, {r['season']}")

add_counts = training[training["change_type"] == CHANGE_TYPE]["pitch_type"].value_counts()
viable_pitch_types = add_counts[add_counts >= MIN_GROUP_SIZE_FOR_REGRESSION].index.tolist()


# ============================================================
# RUN THE IDENTICAL GRID AGAINST REAL AND PLACEBO DATA
# ============================================================

def run_grid(training_data, label):
    print(f"\nRunning grid against {label} synergy data...")
    rows = []
    for _, target_row in target_rows.iterrows():
        target_features = {c: target_row[c] for c in SIMILARITY_FEATURE_COLS}
        target_features["usage_pct_delta"] = 10.0

        target_baseline_rows = outcomes[outcomes["player_id"] == target_row["player_id"]]
        target_outcome_before = (
            target_baseline_rows[OUTCOME_METRIC].iloc[0]
            if len(target_baseline_rows) > 0 and OUTCOME_METRIC in target_baseline_rows.columns
            else training_data[f"{OUTCOME_METRIC}_before"].median()
        )

        for pitch_type in viable_pitch_types:
            result = predict_arsenal_change_effect(
                target_features, pitch_type, CHANGE_TYPE,
                target_outcome_before, OUTCOME_METRIC,
                target_row["player_id"], target_row["season"],
                training_df=training_data
            )

            if result.get("used_synergy_feature"):
                rows.append({
                    "target": f"{target_row['player_name']}_{target_row['season']}",
                    "pitch_type": pitch_type,
                    "synergy_coef": result.get("synergy_coef"),
                })

    return pd.DataFrame(rows)


real_results = run_grid(training, "REAL")
placebo_results = run_grid(training_placebo, "PLACEBO (shuffled)")


# ============================================================
# COMPARE
# ============================================================

print("\n\n==============================")
print("Real vs. Placebo synergy_coef comparison")
print("==============================\n")

for label, df in [("REAL", real_results), ("PLACEBO (shuffled)", placebo_results)]:
    if len(df) == 0:
        print(f"{label}: no rows produced (synergy feature never had enough coverage)")
        continue

    avg_mag = df["synergy_coef"].abs().mean()
    per_target_consistency = df.groupby("target")["synergy_coef"].apply(sign_consistency)
    avg_consistency = per_target_consistency.mean()

    print(f"{label}:")
    print(f"  n_predictions: {len(df)}")
    print(f"  avg |synergy_coef|: {avg_mag:.4f}")
    print(f"  avg sign consistency: {avg_consistency:.1%}")
    print()

if len(real_results) > 0 and len(placebo_results) > 0:
    real_mag = real_results["synergy_coef"].abs().mean()
    placebo_mag = placebo_results["synergy_coef"].abs().mean()
    real_consistency = real_results.groupby("target")["synergy_coef"].apply(sign_consistency).mean()
    placebo_consistency = placebo_results.groupby("target")["synergy_coef"].apply(sign_consistency).mean()

    print("==============================")
    print("Verdict")
    print("==============================\n")

    if placebo_consistency >= real_consistency * 0.8 and placebo_mag >= real_mag * 0.5:
        print(
            "WARNING: the placebo (shuffled, causally meaningless) "
            "synergy feature performs comparably to the real one -- "
            "this is evidence the real feature's apparent strength is "
            "an ARTIFACT (of scale, distribution, or regression "
            "mechanics), NOT genuine pitcher-specific signal. The real "
            "result from Check 4c should NOT be trusted as-is."
        )
    else:
        print(
            "The placebo (shuffled) synergy feature performs "
            "meaningfully WORSE than the real one on both magnitude "
            "and sign consistency -- this VALIDATES that the real "
            "feature's strength comes from genuine pitcher-specific "
            "signal, not an artifact of scale or regression mechanics."
        )
        print(
            f"\nReal:    magnitude={real_mag:.4f}, consistency={real_consistency:.1%}"
        )
        print(
            f"Placebo: magnitude={placebo_mag:.4f}, consistency={placebo_consistency:.1%}"
        )
