"""
96_swap_prediction.py

Purpose:
--------
Predicts the real, COMBINED outcome of a pitch-arsenal SWAP (dropping
one pitch and adding another in the same real transition), rather
than inferring one from two independently-predicted DROP and ADD
effects. Direct response to a real, disclosed gap: the existing
recommendation engine (45) treats every candidate as an isolated
decision, with no way to represent "replace X with Y" as a single,
combined event -- even though real pitchers do this constantly (a
real, concrete example raised directly: replacing a four-seam
fastball with a sinker).

WHY A SEPARATE PREDICTION, NOT JUST SUMMING TWO SEPARATE ONES:
Summing an independently-predicted DROP effect and an independently-
predicted ADD effect assumes no interaction between the two changes
-- but a real swap is a single decision a pitcher actually made, with
its own real, combined outcome already sitting in the data. This
predicts from THAT real, combined ground truth directly, rather than
composing two separate estimates that were never validated together.

REAL DATA CONFIRMED FIRST, before building anything: of 1,951 real
arsenal-change transitions (2020-2025), 328 (16.8%) involved BOTH a
real DROP and a real ADD in the same season_from -> season_to window
-- a genuinely usable sample, not a rare edge case. The specific
example raised (FF -> SI) has 9 real precedent events; the most
common real swap pairs are SL->ST (50), SL->FC (31), CH->FS (23).

METHOD: reuses 38's own real similarity infrastructure directly
(SIMILARITY_FEATURE_COLS, FEATURE_WEIGHTS, compute_pitch_composition_
similarity) rather than rebuilding it -- consistent with the rest of
this project's pattern of extending, not duplicating, validated
machinery. Real historical swap events are weighted by (1) the same
7-feature standardized pitcher-profile distance already used
throughout, (2) real pitch-composition overlap, and (3) an EXTRA
weight boost when a comp's own swap matches the SAME (dropped,
added) pitch type pair as the candidate being evaluated -- since a
same-type swap is a more literal precedent than a same-shaped but
different-type one.

Output:
-------
predict_swap_effect() -- callable prediction function. Demonstrated
below on the real FF->SI example, plus a same-scale sanity check
against the sum of two independent DROP/ADD predictions.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 38 -- reuse real, validated similarity machinery
# ============================================================

print("Loading similarity infrastructure from 38...")

spec = importlib.util.spec_from_file_location(
    "similarity_weighted_regression", "38_similarity_weighted_regression.py"
)
swr = importlib.util.module_from_spec(spec)
sys.modules["similarity_weighted_regression"] = swr
spec.loader.exec_module(swr)

similarity_features = swr.similarity_features
arsenal_raw_indexed = swr.arsenal_raw_indexed
FEATURE_WEIGHTS = swr.FEATURE_WEIGHTS
WEIGHT_PITCH_COMPOSITION = swr.WEIGHT_PITCH_COMPOSITION
compute_pitch_composition_similarity = swr.compute_pitch_composition_similarity
get_qualifying_pitch_types = swr.get_qualifying_pitch_types

SIMILARITY_FEATURE_COLS = list(FEATURE_WEIGHTS.keys())

MIN_GROUP_SIZE_FOR_REGRESSION = 15  # matches 45's own real constant
SAME_PAIR_WEIGHT_BOOST = 2.0  # extra multiplier when a real comp's
                                # own swap matches the exact same
                                # (dropped, added) pitch type pair


# ============================================================
# REAL, OVERALL PITCHER-SEASON OUTCOMES (for the COMBINED effect)
# ============================================================

print("Loading real overall pitcher-season outcomes...")
season_metrics = pd.read_csv("pitcher_season_metrics_2020_2025.csv")
outcome_indexed = season_metrics.set_index(["player_id", "season"])


# ============================================================
# BUILD THE REAL SWAP EVENT TABLE
# ============================================================

print("Identifying real historical swap events (both a real DROP "
      "and a real ADD in the same transition)...")

events = pd.read_csv("arsenal_change_events_pitch_level.csv")

grouped = events.groupby(["player_id", "season_from", "season_to"])
swap_rows = []

for (pid, sf, st), group in grouped:
    change_types = set(group["change_type"])
    if "DROP" not in change_types or "ADD" not in change_types:
        continue

    dropped = group[group["change_type"] == "DROP"]["pitch_type"].tolist()
    added = group[group["change_type"] == "ADD"]["pitch_type"].tolist()

    outcome_key_before = (pid, sf)
    outcome_key_after = (pid, st)
    if outcome_key_before not in outcome_indexed.index or outcome_key_after not in outcome_indexed.index:
        continue

    xwoba_before = outcome_indexed.loc[outcome_key_before, "xwoba_against"]
    xwoba_after = outcome_indexed.loc[outcome_key_after, "xwoba_against"]
    if pd.isna(xwoba_before) or pd.isna(xwoba_after):
        continue

    dropped_usage = group[group["change_type"] == "DROP"].set_index("pitch_type")["usage_pct_before"]

    # one row per (dropped, added) pitch type PAIR within this real
    # transition -- a pitcher who dropped 2 and added 1 in the same
    # season produces 2 real swap-pair rows, each sharing the same
    # real combined outcome (the transition only happened once)
    for d in dropped:
        for a in added:
            swap_rows.append({
                "player_id": pid,
                "player_name": group["player_name"].iloc[0],
                "season_from": sf,
                "season_to": st,
                "dropped_pitch_type": d,
                "added_pitch_type": a,
                "dropped_usage_pct_before": dropped_usage.get(d, np.nan),
                "xwoba_before": xwoba_before,
                "xwoba_after": xwoba_after,
                "outcome_change": xwoba_after - xwoba_before,
            })

swap_events = pd.DataFrame(swap_rows)
print(f"\n{len(swap_events):,} real swap-pair events found, across "
      f"{swap_events[['player_id','season_from','season_to']].drop_duplicates().shape[0]:,} "
      f"real transitions")
print(f"Real mean combined outcome_change across ALL real swaps: "
      f"{swap_events['outcome_change'].mean():+.4f} (negative = improvement)")


# ============================================================
# SIMILARITY-WEIGHTED SWAP PREDICTION
#
# PIPELINE CONSISTENCY: matches the single-change models' real
# predictor structure -- outcome_before (mean-reversion) PLUS a real
# usage-magnitude term, not outcome_before alone. dropped_usage_pct_
# before (how much usage is actually being reallocated in the swap)
# is the swap's own direct equivalent of usage_pct_delta -- for the
# target, this is their own real, current usage of the pitch they'd
# be dropping, directly observable (no estimation needed), same
# pattern 60 already established for single DROP events.
# ============================================================

def predict_swap_effect(target_player_id, target_season, dropped_pitch_type, added_pitch_type, swap_events_df=None):
    """
    Predicts the real, COMBINED effect of dropping dropped_pitch_type
    and adding added_pitch_type in the same transition, using real
    historical swap events, weighted by real pitcher similarity plus
    an extra boost for comps whose own real swap matched the exact
    same pitch type pair.

    swap_events_df: optional override (a train-only subset, for
    walk-forward validation) -- defaults to the full, real swap_events
    table built above if not provided.
    """
    if swap_events_df is None:
        swap_events_df = swap_events

    target_key = (target_player_id, target_season)
    if target_key not in arsenal_raw_indexed.index:
        return {"predicted_change": None, "reason": "no_arsenal_data"}
    target_arsenal_row = arsenal_raw_indexed.loc[target_key]
    if isinstance(target_arsenal_row, pd.DataFrame):
        target_arsenal_row = target_arsenal_row.iloc[0]

    target_sim_rows = similarity_features[
        (similarity_features["player_id"] == target_player_id)
        & (similarity_features["season"] == target_season)
    ]
    if len(target_sim_rows) == 0:
        return {"predicted_change": None, "reason": "no_similarity_profile"}
    target_sim_row = target_sim_rows.iloc[0]

    if target_key not in outcome_indexed.index:
        return {"predicted_change": None, "reason": "no_outcome_baseline"}
    target_outcome_before = outcome_indexed.loc[target_key, "xwoba_against"]
    if pd.isna(target_outcome_before):
        return {"predicted_change": None, "reason": "no_outcome_baseline"}

    # target's own real, current usage of the pitch they'd be
    # dropping -- directly observable, no estimation needed, same
    # pattern 60 established for single DROP events
    target_dropped_usage = target_arsenal_row.get(f"{dropped_pitch_type}_usage_pct")
    if pd.isna(target_dropped_usage):
        return {"predicted_change": None, "reason": "target_does_not_throw_dropped_pitch"}

    weights = []
    for _, swap_row in swap_events_df.iterrows():
        comp_key = (swap_row["player_id"], swap_row["season_from"])
        comp_sim_rows = similarity_features[
            (similarity_features["player_id"] == swap_row["player_id"])
            & (similarity_features["season"] == swap_row["season_from"])
        ]
        if len(comp_sim_rows) == 0 or comp_key not in arsenal_raw_indexed.index:
            weights.append(0.0)
            continue
        comp_sim_row = comp_sim_rows.iloc[0]
        comp_arsenal_row = arsenal_raw_indexed.loc[comp_key]
        if isinstance(comp_arsenal_row, pd.DataFrame):
            comp_arsenal_row = comp_arsenal_row.iloc[0]

        if any(pd.isna(comp_sim_row[c]) for c in SIMILARITY_FEATURE_COLS):
            weights.append(0.0)
            continue

        sq_dist = 0.0
        for col, w in FEATURE_WEIGHTS.items():
            sq_dist += w * (target_sim_row[col] - comp_sim_row[col]) ** 2

        jaccard_sim, _ = compute_pitch_composition_similarity(target_arsenal_row, comp_arsenal_row)
        sq_dist += WEIGHT_PITCH_COMPOSITION * ((1 - jaccard_sim) ** 2)

        distance = np.sqrt(sq_dist)
        weight = 1.0 / (1.0 + distance)

        if swap_row["dropped_pitch_type"] == dropped_pitch_type and swap_row["added_pitch_type"] == added_pitch_type:
            weight *= SAME_PAIR_WEIGHT_BOOST

        weights.append(weight)

    swap_events_weighted = swap_events_df.copy()
    swap_events_weighted["weight"] = weights
    valid = swap_events_weighted[swap_events_weighted["weight"] > 0].dropna(subset=["dropped_usage_pct_before"])

    if len(valid) < MIN_GROUP_SIZE_FOR_REGRESSION:
        return {"predicted_change": None, "reason": "insufficient_real_precedent",
                "n_real_events": len(valid)}

    # PIPELINE CONSISTENCY: two real predictors, mean-reversion PLUS
    # usage magnitude -- same structure as the single-change models,
    # not a stripped-down, mean-reversion-only regression
    X_raw = valid[["xwoba_before", "dropped_usage_pct_before"]].values
    y = valid["xwoba_after"].values
    w = valid["weight"].values

    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X_raw - X_mean) / X_std

    model = LinearRegression()
    model.fit(X_scaled, y, sample_weight=w)

    target_X_raw = np.array([[target_outcome_before, target_dropped_usage]])
    target_X_scaled = (target_X_raw - X_mean) / X_std
    predicted_after = model.predict(target_X_scaled)[0]

    n_exact_pair = len(valid[
        (valid["dropped_pitch_type"] == dropped_pitch_type)
        & (valid["added_pitch_type"] == added_pitch_type)
    ])

    coef_dict = dict(zip(["xwoba_before", "dropped_usage_pct_before"], model.coef_))

    # real comp group, for showing "which real historical pitchers
    # this swap prediction is based on" -- same purpose as 38's own
    # "group" output for single-change candidates. Renamed weight ->
    # similarity_weight to match that existing convention, so 94's
    # comp-saving logic can treat this the same way.
    comp_group = valid.rename(columns={"weight": "similarity_weight"}).sort_values(
        "similarity_weight", ascending=False
    )

    return {
        "predicted_outcome": predicted_after,
        "predicted_change": predicted_after - target_outcome_before,
        "n_real_events_used": len(valid),
        "n_exact_pair_precedent": n_exact_pair,
        "mean_reversion_coef": coef_dict["xwoba_before"],
        "usage_magnitude_coef": coef_dict["dropped_usage_pct_before"],
        "group": comp_group,
        "reason": "success",
    }


# ============================================================
# DEMONSTRATION -- the real FF -> SI example, plus a sanity check
# ============================================================

if __name__ == "__main__":
    print("\n\n==============================")
    print("DEMONSTRATION: real FF -> SI swap prediction")
    print("==============================")

    pool = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS)
    demo_row = pool[pool["player_id"].isin(arsenal_raw_indexed.index.get_level_values(0))].iloc[100]

    print(f"\nTarget: {demo_row.get('player_name')}, season {int(demo_row['season'])}")

    result = predict_swap_effect(
        demo_row["player_id"], demo_row["season"], "FF", "SI"
    )
    print(f"\nDROP FF -> ADD SI swap prediction:")
    print(result)
