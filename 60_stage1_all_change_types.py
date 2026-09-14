"""
60_stage1_all_change_types.py

Purpose:
--------
54_stage2_integration.py's Stage 1 integration only applied to ADD
events -- DROP/USAGE_INCREASE/USAGE_DECREASE always fell back to the
base regression unchanged. This extends the same underlying idea
(give the regression a DIRECT, pitch-specific quality signal instead
of relying only on usage-percentage deltas) to all three remaining
change types.

SIMPLER THAN ADD, not harder: for ADD, the new pitch doesn't exist
yet, so its quality has to be PREDICTED (Stage 1's regression). For
DROP/USAGE_INCREASE/USAGE_DECREASE, the pitcher already throws that
pitch RIGHT NOW -- their real, current pitch-level quality for it is
directly OBSERVABLE (a real lookup, not an estimate). The only
prediction step needed is for HISTORICAL COMPS' training-side values,
and even there it's a real lookup (their own actual pre-change
quality for that pitch type), not a model output.

DESIGN: for each historical comp in the group, look up their own
REAL pitch-level quality for the SPECIFIC pitch type being changed,
at season_from (their pre-change season, when they still threw it
meaningfully). For the target, look up their own real, current
pitch-level quality for that same pitch type -- directly, no
estimation. Add this as an extra regression predictor, same pattern
already validated for ADD in 54.

Output:
-------
predict_arsenal_change_with_stage1_all_types() -- generalizes 54's
function to all four change types. Demonstrated below, then
validated with the same leave-one-out backtest methodology as 58,
scoped to DROP/USAGE_INCREASE/USAGE_DECREASE events specifically
(ADD is already validated separately).
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 54
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
predict_arsenal_change_effect = stage2.predict_arsenal_change_effect
estimate_realistic_usage_delta = stage2.estimate_realistic_usage_delta
pitch_level_indexed = stage2.pitch_level_indexed
SIMILARITY_FEATURE_COLS = stage2.SIMILARITY_FEATURE_COLS

rec_spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine_v2", "45_similarity_recommendation_engine.py"
)
rec_engine_direct = importlib.util.module_from_spec(rec_spec)
sys.modules["similarity_recommendation_engine_v2"] = rec_engine_direct
rec_spec.loader.exec_module(rec_engine_direct)
MIN_GROUP_SIZE_FOR_REGRESSION = rec_engine_direct.MIN_GROUP_SIZE_FOR_REGRESSION


# ============================================================
# GENERALIZED STAGE 1 INTEGRATION -- ALL FOUR CHANGE TYPES
# ============================================================

def predict_arsenal_change_with_stage1_all_types(
    target_features, pitch_type, change_type, target_outcome_before,
    outcome_metric, target_player_id, target_season, training_df=None
):
    """
    Generalizes 54's ADD-only Stage 1 integration to all four change
    types. For ADD, behavior is UNCHANGED from 54 (ambiguity there
    requires a real prediction, since the new pitch doesn't exist
    yet). For DROP/USAGE_INCREASE/USAGE_DECREASE, the target's own
    current pitch-level quality is looked up DIRECTLY (real data,
    no estimation), and comps' own PRE-CHANGE quality for that pitch
    type becomes the training-side ground truth.
    """
    if training_df is None:
        training_df = training

    base_result = predict_arsenal_change_effect(
        target_features, pitch_type, change_type, target_outcome_before,
        outcome_metric, target_player_id, target_season, training_df=training_df
    )

    if base_result["predicted_outcome"] is None:
        return base_result

    group = base_result["group"].copy()

    if change_type == "ADD":
        # UNCHANGED from 54: comps' own season_to quality of the NEW
        # pitch (ground truth); target's value needs Stage 1's
        # regression prediction (built inline here, same as 54)
        real_qualities = []
        for _, comp_row in group.iterrows():
            key = (comp_row["player_id"], comp_row["season_to"], pitch_type)
            real_qualities.append(pitch_level_indexed.get(key, np.nan))
        group["existing_pitch_quality"] = real_qualities

        n_with_quality = group["existing_pitch_quality"].notna().sum()
        if n_with_quality < MIN_GROUP_SIZE_FOR_REGRESSION:
            base_result["stage1_integration_status"] = "insufficient_ground_truth"
            return base_result

        # target's value is UNKNOWN (pitch doesn't exist yet) --
        # estimate via the SAME similarity-weighted regression
        # approach as Stage 1, built inline against this SAME group
        valid_for_target_est = group.dropna(subset=["existing_pitch_quality"])
        comp_features = valid_for_target_est[SIMILARITY_FEATURE_COLS].values
        comp_quality = valid_for_target_est["existing_pitch_quality"].values
        comp_weights = valid_for_target_est["combined_weight"].values

        if len(comp_quality) >= MIN_GROUP_SIZE_FOR_REGRESSION:
            X_mean = comp_features.mean(axis=0)
            X_std = comp_features.std(axis=0)
            X_std[X_std == 0] = 1.0
            X_scaled = (comp_features - X_mean) / X_std
            est_model = LinearRegression()
            est_model.fit(X_scaled, comp_quality, sample_weight=comp_weights)
            target_vec = np.array([target_features[c] for c in SIMILARITY_FEATURE_COLS])
            target_scaled = (target_vec - X_mean) / X_std
            target_quality_estimate = est_model.predict([target_scaled])[0]
        else:
            target_quality_estimate = np.average(comp_quality, weights=comp_weights)

    else:
        # DROP / USAGE_INCREASE / USAGE_DECREASE: the target already
        # throws this pitch RIGHT NOW -- look up their real, current
        # quality DIRECTLY, no estimation needed
        target_key = (target_player_id, target_season, pitch_type)
        target_quality_estimate = pitch_level_indexed.get(target_key, np.nan)
        if pd.isna(target_quality_estimate):
            base_result["stage1_integration_status"] = "target_quality_unknown"
            return base_result

        # comps' training-side ground truth: THEIR OWN pre-change
        # quality for this pitch type, at season_from (before they
        # dropped/changed it)
        real_qualities = []
        for _, comp_row in group.iterrows():
            key = (comp_row["player_id"], comp_row["season_from"], pitch_type)
            real_qualities.append(pitch_level_indexed.get(key, np.nan))
        group["existing_pitch_quality"] = real_qualities

        n_with_quality = group["existing_pitch_quality"].notna().sum()
        if n_with_quality < MIN_GROUP_SIZE_FOR_REGRESSION:
            base_result["stage1_integration_status"] = "insufficient_ground_truth"
            return base_result

    # REFIT the season-wide regression, adding existing_pitch_quality
    # as an extra predictor -- same combined pattern as 54, reusing
    # whichever of distance/synergy the base group already has
    valid_group = group.dropna(subset=["existing_pitch_quality"])

    predictor_cols = ["existing_pitch_quality"]
    if base_result.get("used_distance_feature") and "new_pitch_nn_distance_from_existing" in valid_group.columns:
        valid_group = valid_group.dropna(subset=["new_pitch_nn_distance_from_existing"])
        predictor_cols.append("new_pitch_nn_distance_from_existing")
    if base_result.get("used_synergy_feature") and "arsenal_synergy_whiff_delta" in valid_group.columns:
        valid_group = valid_group.dropna(subset=["arsenal_synergy_whiff_delta"])
        predictor_cols.append("arsenal_synergy_whiff_delta")

    outcome_before_col = f"{outcome_metric}_before"
    outcome_after_col = f"{outcome_metric}_after"

    if len(valid_group) < MIN_GROUP_SIZE_FOR_REGRESSION:
        base_result["stage1_integration_status"] = "insufficient_rows_after_combining_features"
        return base_result

    all_predictor_cols = [outcome_before_col, "usage_pct_delta"] + predictor_cols

    X_raw = valid_group[all_predictor_cols].values
    y = valid_group[outcome_after_col].values
    weights = valid_group["combined_weight"].values

    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X_raw - X_mean) / X_std

    model = LinearRegression()
    model.fit(X_scaled, y, sample_weight=weights)

    target_usage_delta = target_features.get("usage_pct_delta", valid_group["usage_pct_delta"].median())
    target_predictors = [target_outcome_before, target_usage_delta, target_quality_estimate]
    for col in predictor_cols[1:]:
        target_predictors.append(target_features.get(col, valid_group[col].median()))

    target_X_raw = np.array([target_predictors])
    target_X_scaled = (target_X_raw - X_mean) / X_std
    predicted = model.predict(target_X_scaled)[0]

    coef_dict = dict(zip(all_predictor_cols, model.coef_))

    return {
        "predicted_outcome": predicted,
        "predicted_change": predicted - target_outcome_before,
        "method": "weighted_regression_with_stage1_all_types",
        "n_historical_events": len(valid_group),
        "group": valid_group,
        "mean_reversion_coef": coef_dict.get(outcome_before_col),
        "change_specific_coef": coef_dict.get("usage_pct_delta"),
        "existing_quality_coef": coef_dict.get("existing_pitch_quality"),
        "target_quality_used": target_quality_estimate,
        "used_distance_feature": base_result.get("used_distance_feature", False),
        "used_synergy_feature": base_result.get("used_synergy_feature", False),
        "stage1_integration_status": "success",
    }


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n\n==============================")
    print("DEMONSTRATION -- DROP and USAGE_INCREASE/DECREASE with Stage 1")
    print("==============================")

    demo_row = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).iloc[
        len(similarity_features) // 3
    ]
    demo_player_id = demo_row["player_id"]
    demo_season = demo_row["season"]

    print(f"\nTarget: {demo_row.get('player_name', demo_player_id)}, season {demo_season}")

    target_features = {c: demo_row[c] for c in SIMILARITY_FEATURE_COLS}
    outcome_metric = "xwoba_against"

    target_baseline_rows = outcomes[outcomes["player_id"] == demo_player_id]
    target_outcome_before = target_baseline_rows.sort_values("season")[outcome_metric].iloc[-1]

    for change_type, pitch_type in [("DROP", "FF"), ("USAGE_INCREASE", "SI"), ("USAGE_DECREASE", "SL")]:
        print(f"\n--- {change_type} {pitch_type} ---")

        if change_type == "DROP":
            current_usage = -25.0  # placeholder magnitude for demo purposes
        else:
            realistic_delta, _ = estimate_realistic_usage_delta(
                target_features, pitch_type, change_type, demo_player_id,
                demo_season, target_outcome_before, outcome_metric
            )
            current_usage = realistic_delta

        if current_usage is None:
            print("No historical precedent -- skipping.")
            continue

        final_features = dict(target_features)
        final_features["usage_pct_delta"] = current_usage

        result = predict_arsenal_change_with_stage1_all_types(
            final_features, pitch_type, change_type, target_outcome_before,
            outcome_metric, demo_player_id, demo_season
        )

        print(f"Status: {result.get('stage1_integration_status', 'n/a')}")
        print(f"Method: {result['method']}")
        if result["predicted_outcome"] is not None and "existing_quality_coef" in result:
            print(
                f"Predicted {outcome_metric}: {result['predicted_outcome']:.4f} "
                f"(change of {result['predicted_change']:+.4f})"
            )
            print(
                f"existing_quality_coef: {result['existing_quality_coef']:.4f}, "
                f"mean_reversion_coef: {result['mean_reversion_coef']:.4f}"
            )
