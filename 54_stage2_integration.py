"""
54_stage2_integration.py

Purpose:
--------
Wires Stage 1's now-validated pitch-quality prediction (50, redesigned
to an actual weighted regression, confirmed to differentiate by
pitcher and pass its placebo test with real correlation 0.448 vs.
placebo -0.122) into the season-wide regression as a genuinely new,
sharper predictor -- the actual payoff this whole two-stage effort
was building toward.

ARCHITECTURAL CONSTRAINT, worth stating plainly: this canNOT be
built by modifying 38_similarity_weighted_regression.py directly.
50_stage1_pitch_quality_model.py imports FROM 38 (via 45) to build
its own comparison groups -- if 38 tried to import Stage 1's
prediction function back from 50, that would be a circular import
Python cannot resolve. This script sits as a NEW layer ABOVE both,
importing each independently, rather than trying to weave Stage 1
into 38's own internals.

DESIGN: for each row in the comparison group (real historical
pitchers who ALREADY added this pitch type), their pitch-level
quality is ALREADY KNOWN -- no need to predict what we already have
ground truth for. Stage 1's prediction is used ONLY for the TARGET,
whose new pitch hasn't been thrown yet. This is the exact same
pattern already used for the distance and synergy features (real
values for training rows, estimated value for the target) -- Stage 1
quality is just one more feature following that same, already-
validated pattern, not a new architecture.

The group's own new_pitch_nn_distance_from_existing and arsenal_
synergy_whiff_delta columns (already computed by 38's own group-
building step) are REUSED directly rather than recomputed, so this
integration adds Stage 1 alongside the existing features rather than
replacing them.

Output:
-------
predict_arsenal_change_with_stage1() -- a reusable wrapper function,
demonstrated below on a real target pitcher/pitch-type combination.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 50 (Stage 1) AND 45/38 (Stage 2 base) INDEPENDENTLY
# ============================================================

print("Loading Stage 1 (50) and base Stage 2 (45/38) machinery...")

stage1_spec = importlib.util.spec_from_file_location(
    "stage1_pitch_quality_model", "50_stage1_pitch_quality_model.py"
)
stage1 = importlib.util.module_from_spec(stage1_spec)
sys.modules["stage1_pitch_quality_model"] = stage1
stage1_spec.loader.exec_module(stage1)

training = stage1.training
similarity_features = stage1.similarity_features
outcomes = stage1.outcomes
predict_arsenal_change_effect = stage1.predict_arsenal_change_effect
predict_new_pitch_quality = stage1.predict_new_pitch_quality
estimate_realistic_usage_delta = stage1.estimate_realistic_usage_delta
pitch_level_indexed = stage1.pitch_level_indexed
SIMILARITY_FEATURE_COLS = stage1.SIMILARITY_FEATURE_COLS

# MIN_GROUP_SIZE_FOR_REGRESSION is a LOCAL variable inside 50's own
# predict_new_pitch_quality function, not a module-level constant --
# not accessible via stage1.MIN_GROUP_SIZE_FOR_REGRESSION. Get it
# from a direct import of 45 instead, where it genuinely IS a
# module-level constant (same fix pattern already used in
# 53_new_pitch_shape_differentiation_check.py for a similar issue).
rec_spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine_v2", "45_similarity_recommendation_engine.py"
)
rec_engine_direct = importlib.util.module_from_spec(rec_spec)
sys.modules["similarity_recommendation_engine_v2"] = rec_engine_direct
rec_spec.loader.exec_module(rec_engine_direct)
MIN_GROUP_SIZE_FOR_REGRESSION = rec_engine_direct.MIN_GROUP_SIZE_FOR_REGRESSION


# ============================================================
# STAGE 2 INTEGRATION FUNCTION
# ============================================================

def predict_arsenal_change_with_stage1(
    target_features, pitch_type, change_type, target_outcome_before,
    outcome_metric, target_player_id, target_season, training_df=None
):
    """
    Wraps predict_arsenal_change_effect (38), adding Stage 1's
    predicted pitch quality as an additional regression predictor --
    ADD events only, since Stage 1 only applies to newly-added
    pitches. All other change types delegate unchanged to the
    original function.
    """
    if training_df is None:
        training_df = training

    # non-ADD events: Stage 1 doesn't apply, use 38's existing
    # behavior completely unchanged
    if change_type != "ADD":
        return predict_arsenal_change_effect(
            target_features, pitch_type, change_type, target_outcome_before,
            outcome_metric, target_player_id, target_season, training_df=training_df
        )

    # get the BASE comparison group from 38's existing, unmodified
    # machinery -- reused directly, not rebuilt
    base_result = predict_arsenal_change_effect(
        target_features, pitch_type, change_type, target_outcome_before,
        outcome_metric, target_player_id, target_season, training_df=training_df
    )

    if base_result["predicted_outcome"] is None:
        return base_result  # can't proceed -- same failure 38 already reports

    group = base_result["group"].copy()

    # for each historical comp, look up their REAL, ACTUAL pitch-
    # level quality -- ground truth, not a prediction, since these
    # are real events that already happened
    real_qualities = []
    for _, comp_row in group.iterrows():
        key = (comp_row["player_id"], comp_row["season_to"], pitch_type)
        real_qualities.append(pitch_level_indexed.get(key, np.nan))
    group["stage1_pitch_quality"] = real_qualities

    n_with_quality = group["stage1_pitch_quality"].notna().sum()

    if n_with_quality < MIN_GROUP_SIZE_FOR_REGRESSION:
        # not enough real ground truth to add this predictor safely --
        # fall back to the base result unchanged, same graceful-
        # degradation pattern used throughout this project
        base_result["stage1_integration_status"] = "insufficient_ground_truth"
        return base_result

    # TARGET's own value is UNKNOWN (their new pitch hasn't been
    # thrown) -- this is where Stage 1's actual prediction is used,
    # the one genuinely new piece of information this integration adds
    stage1_result = predict_new_pitch_quality(
        target_player_id, target_season, pitch_type, training_df=training_df
    )
    if stage1_result["predicted_pitch_quality"] is None:
        base_result["stage1_integration_status"] = "stage1_prediction_failed"
        return base_result

    # build the combined predictor set -- REUSE whichever of
    # distance/synergy the base group already has (computed by 38's
    # own group-building step), plus the new stage1_pitch_quality
    valid_group = group.dropna(subset=["stage1_pitch_quality"])

    predictor_cols = ["stage1_pitch_quality"]
    target_extra_values = {"stage1_pitch_quality": stage1_result["predicted_pitch_quality"]}

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
    target_predictors = [target_outcome_before, target_usage_delta]
    for col in predictor_cols:
        if col == "stage1_pitch_quality":
            target_predictors.append(target_extra_values["stage1_pitch_quality"])
        else:
            target_predictors.append(target_features.get(col, valid_group[col].median()))

    target_X_raw = np.array([target_predictors])
    target_X_scaled = (target_X_raw - X_mean) / X_std
    predicted = model.predict(target_X_scaled)[0]

    coef_dict = dict(zip(all_predictor_cols, model.coef_))

    return {
        "predicted_outcome": predicted,
        "predicted_change": predicted - target_outcome_before,
        "method": "weighted_regression_with_stage1",
        "n_historical_events": len(valid_group),
        "group": valid_group,
        "mean_reversion_coef": coef_dict.get(outcome_before_col),
        "change_specific_coef": coef_dict.get("usage_pct_delta"),
        "stage1_quality_coef": coef_dict.get("stage1_pitch_quality"),
        "new_pitch_distance_coef": coef_dict.get("new_pitch_nn_distance_from_existing"),
        "synergy_coef": coef_dict.get("arsenal_synergy_whiff_delta"),
        "stage1_predicted_quality_used": stage1_result["predicted_pitch_quality"],
        # propagated from base_result -- REQUIRED for compatibility
        # with 45's existing fallback_values_used disclosure logic,
        # which checks these two keys directly
        "used_distance_feature": base_result.get("used_distance_feature", False),
        "used_synergy_feature": base_result.get("used_synergy_feature", False),
        "stage1_integration_status": "success",
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

    target_features = {c: demo_row[c] for c in SIMILARITY_FEATURE_COLS}
    outcome_metric = "xwoba_against"

    target_baseline_rows = outcomes[outcomes["player_id"] == demo_player_id]
    target_outcome_before = target_baseline_rows.sort_values("season")[outcome_metric].iloc[-1]

    for pitch_type in ["SL", "ST", "FS"]:
        print(f"\n--- ADD {pitch_type}, WITH Stage 1 integration ---")

        realistic_delta, _ = estimate_realistic_usage_delta(
            target_features, pitch_type, "ADD", demo_player_id,
            demo_season, target_outcome_before, outcome_metric
        )
        if realistic_delta is None:
            print("No historical precedent -- skipping.")
            continue

        final_features = dict(target_features)
        final_features["usage_pct_delta"] = realistic_delta

        result = predict_arsenal_change_with_stage1(
            final_features, pitch_type, "ADD", target_outcome_before,
            outcome_metric, demo_player_id, demo_season
        )

        print(f"Status: {result.get('stage1_integration_status', 'n/a (non-ADD or base fallback)')}")
        print(f"Method: {result['method']}")
        if result["predicted_outcome"] is not None:
            print(
                f"Predicted {outcome_metric}: {result['predicted_outcome']:.4f} "
                f"(change of {result['predicted_change']:+.4f})"
            )
            if "stage1_quality_coef" in result and result["stage1_quality_coef"] is not None:
                print(
                    f"stage1_quality_coef: {result['stage1_quality_coef']:.4f}, "
                    f"mean_reversion_coef: {result['mean_reversion_coef']:.4f}, "
                    f"change_specific_coef: {result['change_specific_coef']:.4f}"
                )
                print(f"Stage 1 predicted quality used for target: {result['stage1_predicted_quality_used']:.4f}")
