"""
50_stage1_pitch_quality_model.py

Purpose:
--------
Stage 1 of the two-stage model discussed several turns back: predict
a SPECIFIC new pitch's OWN quality (pitch-level xwoba_against, from
47_pull_pitch_level_outcomes.py, already MIN_PITCHES=50-filtered),
rather than only ever predicting a whole SEASON's blended outcome.
This is the real fix for the mean-reversion-dominance problem this
whole thread kept running into -- a pitch-specific target isn't
competing against everything else that happened that season the way
the current season-wide target does.

DESIGN, REUSING EXISTING MACHINERY RATHER THAN REBUILDING IT:
predict_arsenal_change_effect() (38) already builds a similarity-
AND-reliability-weighted comparison group for any (pitch_type,
"ADD") search -- computed from the SAME validated weighting this
whole project has already built and placebo-tested. Stage 1 reuses
that EXACT group, rather than reimplementing its own separate
similarity computation. Instead of looking at that group's SEASON-
WIDE predicted outcome, it looks up each comparison pitcher's OWN
new pitch's PITCH-LEVEL quality (from 47's data, joined on player_
id + season_to + pitch_type), and takes the SAME similarity-
weighted MEDIAN already used elsewhere in this project (45's
weighted_median -- robust to outliers, the same real fix applied to
the usage-delta estimation after the Houck case) as the Stage 1
prediction.

DELIBERATELY STANDALONE FOR NOW: this is built and tested as its
OWN independent piece, NOT yet wired into Stage 2 (the season-wide
regression). Every other feature in this project (distance, synergy)
was validated on its own -- including a placebo test -- before being
folded into the main regression. Stage 1 should get the same
treatment before Stage 2 integration, not be integrated blind.

Output:
-------
predict_new_pitch_quality() -- a reusable function, demonstrated
below on a real target pitcher/pitch-type combination.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 45 (which itself imports from 38)
# ============================================================

print("Loading training data and prediction functions from 45...")

spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine", "45_similarity_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(spec)
sys.modules["similarity_recommendation_engine"] = rec_engine
spec.loader.exec_module(rec_engine)

similarity_features = rec_engine.similarity_features
outcomes = rec_engine.outcomes
training = rec_engine.training
predict_arsenal_change_effect = rec_engine.predict_arsenal_change_effect
estimate_realistic_usage_delta = rec_engine.estimate_realistic_usage_delta
weighted_median = rec_engine.weighted_median
SIMILARITY_FEATURE_COLS = rec_engine.SIMILARITY_FEATURE_COLS


# ============================================================
# SETTINGS
# ============================================================

PITCH_LEVEL_OUTCOMES_FILE = "pitcher_pitch_level_outcomes_2020_2025.csv"
PITCH_QUALITY_METRIC = "xwoba_against"  # from the pitch-level file --
                                          # NOT the season-wide metric
                                          # of the same name


# ============================================================
# LOAD PITCH-LEVEL OUTCOMES
# ============================================================

print(f"\nLoading pitch-level outcomes ({PITCH_LEVEL_OUTCOMES_FILE})...")

pitch_level = pd.read_csv(PITCH_LEVEL_OUTCOMES_FILE)
pitch_level_indexed = pitch_level.set_index(["player_id", "season", "pitch_type"])[PITCH_QUALITY_METRIC]

# GUARDRAIL: a linear regression (unlike the old weighted-median
# approach) can extrapolate PAST the range of real observed values --
# confirmed directly on real data: Ben Rowen (an extreme low-velocity
# outlier, max_velo_z=-5.60) got a predicted FF quality of -0.036, a
# physically impossible negative xwoba_against. A median could never
# produce this (it's always one of the real observed values or
# between two of them); a regression can. Clip predictions to the
# REAL observed range in the filtered pitch-level data -- a data-
# driven bound, not an arbitrary guessed one.
PITCH_QUALITY_MIN = pitch_level[PITCH_QUALITY_METRIC].min()
PITCH_QUALITY_MAX = pitch_level[PITCH_QUALITY_METRIC].max()
print(f"Prediction bounds (real observed range): [{PITCH_QUALITY_MIN:.3f}, {PITCH_QUALITY_MAX:.3f}]")

print(f"Loaded {len(pitch_level):,} pitch-type-level rows (already MIN_PITCHES-filtered)")


# ============================================================
# STAGE 1 PREDICTION FUNCTION
# ============================================================

def predict_new_pitch_quality(target_player_id, target_season, pitch_type, training_df=None):
    """
    Predicts a hypothetical new pitch's OWN quality for the target
    pitcher, by reusing 38's existing similarity-weighted comparison
    group (via predict_arsenal_change_effect) and looking up each
    comp's OWN pitch-level outcome for that same pitch type, instead
    of their season-wide blended outcome.

    training_df: optional override, passed through to the FINAL
    predict_arsenal_change_effect call below -- the step where the
    comparison group used for the pitch-quality lookup is actually
    built. Used by 51_stage1_validation.py for a leave-one-out
    validation (excluding a specific historical event from its own
    comparison group before predicting it).

    HONEST LIMITATION: estimate_realistic_usage_delta() (used just
    below to estimate usage_pct_delta) does NOT accept a training_df
    override -- its own internal comparison group always draws from
    the FULL training set, even when a custom training_df is passed
    here. Disclosed rather than hidden: usage_pct_delta doesn't
    touch pitch-level xwoba_against values directly, so this is a
    less consequential leak than the pitch-quality lookup itself
    would be, but it isn't a textbook-perfect exclusion either.

    Returns dict with predicted_pitch_quality, n_comps_with_pitch_
    level_data (the REAL sample size backing this specific
    prediction -- smaller than the season-wide group size, since not
    every historical comp has a valid pitch-level row after the
    MIN_PITCHES=50 filter), and the underlying group for inspection.
    """
    if training_df is None:
        training_df = training

    target_sim_rows = similarity_features[
        (similarity_features["player_id"] == target_player_id)
        & (similarity_features["season"] == target_season)
    ]
    if len(target_sim_rows) == 0:
        return {"predicted_pitch_quality": None, "method": "no_similarity_profile"}

    target_sim_row = target_sim_rows.iloc[0]
    target_features = {c: target_sim_row[c] for c in SIMILARITY_FEATURE_COLS}

    outcome_metric = "xwoba_against"  # season-wide metric, needed
                                        # only because predict_
                                        # arsenal_change_effect
                                        # requires one -- Stage 1
                                        # does NOT use its season-
                                        # wide prediction, only the
                                        # weighted GROUP it builds
    target_baseline_rows = outcomes[outcomes["player_id"] == target_player_id]
    if len(target_baseline_rows) == 0 or outcome_metric not in target_baseline_rows.columns:
        return {"predicted_pitch_quality": None, "method": "no_outcome_baseline"}
    target_outcome_before = target_baseline_rows.sort_values("season")[outcome_metric].iloc[-1]

    realistic_delta, _ = estimate_realistic_usage_delta(
        target_features, pitch_type, "ADD", target_player_id,
        target_season, target_outcome_before, outcome_metric
    )
    if realistic_delta is None:
        return {"predicted_pitch_quality": None, "method": "no_historical_precedent"}

    final_features = dict(target_features)
    final_features["usage_pct_delta"] = realistic_delta

    # this call's OWN season-wide prediction is NOT used -- only the
    # similarity-and-reliability-weighted GROUP it builds internally.
    # training_df is threaded through HERE specifically, since this
    # is where the group used for the pitch-quality lookup is built
    season_wide_result = predict_arsenal_change_effect(
        final_features, pitch_type, "ADD",
        target_outcome_before, outcome_metric,
        target_player_id, target_season,
        training_df=training_df
    )

    if season_wide_result["predicted_outcome"] is None:
        return {"predicted_pitch_quality": None, "method": "no_usable_group"}

    group = season_wide_result["group"]

    # STAGE 1 REDESIGN: previously took a weighted MEDIAN of comps'
    # own pitch-level quality -- a pure summary of the group's
    # central tendency, with no way to reflect where the TARGET
    # specifically sits within that group. Confirmed directly (52_
    # stage1_pitcher_differentiation_check.py, real data): between-
    # pitcher variance was ~4x SMALLER than between-pitch-type
    # variance -- completely different pitchers adding the same
    # pitch type got nearly identical predictions.
    #
    # FIX: a weighted REGRESSION -- comps' own similarity features
    # as X, their own pitch quality as y, weighted by combined_
    # weight (the same weighting already used for everything else),
    # then predict using the TARGET's own specific feature values.
    # This lets the model learn how quality varies ALONG the
    # similarity dimensions instead of collapsing to the group's
    # center regardless of where the target sits.
    #
    # Design settled via explicit discussion: per-pitch-type (not
    # pooled across types), plain linear (no regularization, matching
    # 38's approach), existing-profile predictors only (new-pitch-
    # shape predictors tested and SCRAPPED -- 53_new_pitch_shape_
    # differentiation_check.py found shape estimates collapse the
    # SAME way quality did, 31-75x smaller between-pitcher variance).
    MIN_GROUP_SIZE_FOR_REGRESSION = 40  # SAME threshold already
                                          # established in 38, reused
                                          # for consistency -- guards
                                          # against fitting 7
                                          # predictors on too few rows

    comp_features = []
    pitch_qualities = []
    weights = []
    for _, comp_row in group.iterrows():
        key = (comp_row["player_id"], comp_row["season_to"], pitch_type)
        quality = pitch_level_indexed.get(key, np.nan)
        if pd.notna(quality):
            comp_features.append([comp_row[c] for c in SIMILARITY_FEATURE_COLS])
            pitch_qualities.append(quality)
            weights.append(comp_row["combined_weight"])

    if len(pitch_qualities) == 0:
        return {
            "predicted_pitch_quality": None,
            "method": "no_comps_with_pitch_level_data",
            "n_comps_with_pitch_level_data": 0,
            "group": group,
        }

    comp_features = np.array(comp_features)
    pitch_qualities_arr = np.array(pitch_qualities)
    weights_arr = np.array(weights)

    if len(pitch_qualities) >= MIN_GROUP_SIZE_FOR_REGRESSION:
        X_mean = comp_features.mean(axis=0)
        X_std = comp_features.std(axis=0)
        X_std[X_std == 0] = 1.0  # avoid divide-by-zero for a
                                   # degenerate no-variance column
        X_scaled = (comp_features - X_mean) / X_std

        model = LinearRegression()
        model.fit(X_scaled, pitch_qualities_arr, sample_weight=weights_arr)

        target_vec = np.array([target_features[c] for c in SIMILARITY_FEATURE_COLS])
        target_scaled = (target_vec - X_mean) / X_std
        predicted_quality = model.predict([target_scaled])[0]

        method = "weighted_regression"
    else:
        # too few comps for a stable 7-predictor fit -- fall back to
        # the ORIGINAL, already-tested weighted-median approach
        # (graceful degradation, same pattern 38 itself uses when a
        # group is too small for its own regression)
        predicted_quality = weighted_median(pitch_qualities_arr, weights_arr)
        method = "weighted_median_fallback"

    # apply the guardrail -- harmless no-op for the median path
    # (mathematically guaranteed to already fall in-range), the real
    # protection for the regression path
    was_clipped = predicted_quality < PITCH_QUALITY_MIN or predicted_quality > PITCH_QUALITY_MAX
    predicted_quality_clipped = float(np.clip(predicted_quality, PITCH_QUALITY_MIN, PITCH_QUALITY_MAX))

    return {
        "predicted_pitch_quality": predicted_quality_clipped,
        "predicted_pitch_quality_unclipped": predicted_quality,
        "was_clipped": was_clipped,
        "method": method,
        "n_comps_with_pitch_level_data": len(pitch_qualities),
        "n_comps_in_full_group": len(group),
        "group": group,
    }


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n\n==============================")
    print("DEMONSTRATION")
    print("==============================")

    demo_row = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).iloc[
        len(similarity_features) // 3
    ]
    demo_player_id = demo_row["player_id"]
    demo_season = demo_row["season"]

    print(f"\nTarget: {demo_row.get('player_name', demo_player_id)}, season {demo_season}")

    for pitch_type in ["SL", "ST", "FS"]:
        print(f"\n--- Predicting new-pitch quality: ADD {pitch_type} ---")
        result = predict_new_pitch_quality(demo_player_id, demo_season, pitch_type)
        print(f"Method: {result['method']}")
        if result["predicted_pitch_quality"] is not None:
            print(
                f"Predicted pitch-level xwoba_against for this new "
                f"{pitch_type}: {result['predicted_pitch_quality']:.3f}"
            )
            print(
                f"Based on {result['n_comps_with_pitch_level_data']} of "
                f"{result['n_comps_in_full_group']} comps in the full "
                f"weighted group (the rest lacked a valid pitch-level "
                f"row after the MIN_PITCHES=50 filter -- a real, "
                f"disclosed sample-size cost of this more granular "
                f"target, same tradeoff as everywhere else this floor "
                f"has been applied)."
            )
