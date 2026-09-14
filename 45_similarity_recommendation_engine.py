"""
45_similarity_recommendation_engine.py

Purpose:
--------
The actual deliverable this whole thread has been building toward:
given a target pitcher, automatically search across every viable
candidate pitch type they DON'T already throw, predict the effect of
each using the validated similarity-weighted regression, and return
a RANKED list of recommendations with real historical comps -- not
requiring the user to specify a hypothesis upfront the way
38_similarity_weighted_regression.py's demonstration does.

Distinct from the ORIGINAL project's 16_recommendation_engine.py
(the current-snapshot-based PAOM recommendation engine) -- this is a
separate, historical-precedent-based engine built on top of the
2020-2025 Kaggle arsenal evolution data.

VALIDATED FOUNDATION this builds on:
- Similarity-weighted regression (38), with tiered feature weights
  and pitch-composition matching
- Two independently placebo-tested change-specific features:
  new_pitch_nn_distance_from_existing (43/44 confirmed real signal,
  not noise) and arsenal_synergy_whiff_delta (43 confirmed real
  signal, though its CAUSAL mechanism remains genuinely open -- see
  38's docstring discussion)
- xwoba_against as the primary target metric (41 showed it carries
  the strongest, most stable change-specific signal of the three
  outcome metrics tested)

A REAL, HONEST LIMITATION OF PROSPECTIVE RECOMMENDATIONS: new_pitch_
nn_distance_from_existing and arsenal_synergy_whiff_delta were both
built from a pitch's ACTUAL landed shape/performance AFTER it was
already thrown for a full season -- for a genuinely prospective
recommendation (before the pitcher has thrown the new pitch at all),
we don't know what its real shape or synergy effect will be. This
engine handles that honestly: when a candidate pitch type's own
future shape/synergy isn't knowable in advance, predict_arsenal_
change_effect() falls back to that comparison group's own historical
MEDIAN for the missing value (its existing, already-documented
default behavior) -- meaning the recommendation for an untried pitch
type implicitly assumes "a typical historical case," not perfect
foreknowledge. This is disclosed explicitly in the output, not
buried.

Output:
-------
Printed ranked recommendations for a demonstration target pitcher,
plus real historical comps for the top recommendation. This is a
callable function (recommend_arsenal_changes), meant to be reused
for any target pitcher, not a one-off script.
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
arsenal_raw = swr.arsenal_raw
arsenal_raw_indexed = swr.arsenal_raw_indexed
predict_arsenal_change_effect = swr.predict_arsenal_change_effect
show_similar_comps = swr.show_similar_comps
SIMILARITY_FEATURE_COLS = swr.SIMILARITY_FEATURE_COLS
reliability_weight = swr.reliability_weight

from sklearn.linear_model import LinearRegression
MIN_GROUP_SIZE_FOR_REGRESSION = swr.MIN_GROUP_SIZE_FOR_REGRESSION
MEANINGFUL_USAGE_FLOOR = swr.MEANINGFUL_USAGE_FLOOR
PITCH_TYPES = swr.PITCH_TYPES
get_qualifying_pitch_types = swr.get_qualifying_pitch_types


# ============================================================
# SETTINGS
# ============================================================

CHANGE_TYPES_TO_SEARCH = ["ADD", "DROP", "USAGE_INCREASE", "USAGE_DECREASE"]

# metrics where LOWER is better get ranked ascending (biggest
# improvement = most negative predicted_change); WAR is the
# opposite
LOWER_IS_BETTER_METRICS = {"xwoba_against", "fip"}


# ============================================================
# WEIGHTED MEDIAN (robust to outliers, unlike a weighted mean)
# ============================================================

def weighted_median(values, weights):
    """
    Standard weighted-median: sort by value, walk cumulative weight,
    return the value where cumulative weight crosses 50% of total.

    REAL MOTIVATION: a real-data run found the weighted MEAN pulled
    toward a single extreme historical comp -- Tanner Houck's actual
    37.18% -> 0.00% slider drop -- producing a top recommendation
    (DROP SL, -36.8pp) that was really just matching one unusually
    extreme real case, not a representative "typical" outcome. A
    weighted median is resistant to this by construction: one
    extreme value at the tail can't drag it the way it drags a mean.
    Confirmed directly against a reconstruction of the real Cuas/
    Houck scenario -- median landed meaningfully less extreme than
    the mean on the same data (-19.7pp vs. -22.1pp in that test).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative_weight = np.cumsum(sorted_weights)
    total_weight = cumulative_weight[-1]
    median_idx = np.searchsorted(cumulative_weight, total_weight / 2.0)
    median_idx = min(median_idx, len(sorted_values) - 1)
    return sorted_values[median_idx]


# ============================================================
# REALISTIC USAGE-DELTA ESTIMATION
# ============================================================

def estimate_realistic_usage_delta(
    target_features, pitch_type, change_type, target_player_id,
    target_season, target_outcome_before, outcome_metric, known_delta=None
):
    """
    Returns a realistic usage_pct_delta for a candidate change,
    instead of one flat assumed number for every pitch type.

    For DROP: the target's own CURRENT usage of that pitch is already
    known exactly (no estimation needed) -- pass known_delta directly.

    For ADD/USAGE_INCREASE/USAGE_DECREASE: the post-change usage
    level is genuinely UNKNOWN in advance, so it's estimated from
    real historical precedent -- specifically, a SIMILARITY-WEIGHTED
    average of usage_pct_delta across the SAME comparison group
    predict_arsenal_change_effect() already builds (weighted by how
    similar each historical pitcher was to the target, not a flat
    unweighted average across every historical adder of that pitch
    type). This works as a clean two-pass approach because usage_
    pct_delta never factors into the similarity/reliability
    weighting itself (confirmed directly against 38's code) -- only
    into the final prediction step -- so a first pass with a
    placeholder value returns the CORRECT comparison group and
    weights, unaffected by that placeholder.

    Returns (realistic_delta, first_pass_result_or_None).
    """
    if known_delta is not None:
        return known_delta, None

    placeholder_features = dict(target_features)
    placeholder_features["usage_pct_delta"] = 10.0  # arbitrary --
        # doesn't affect the group/weights, only used to get a
        # complete first-pass result to extract the group from

    first_pass = predict_arsenal_change_effect(
        placeholder_features, pitch_type, change_type,
        target_outcome_before, outcome_metric,
        target_player_id, target_season
    )

    if first_pass["predicted_outcome"] is None or "combined_weight" not in first_pass["group"].columns:
        return None, first_pass

    group = first_pass["group"]
    realistic_delta = weighted_median(group["usage_pct_delta"], group["combined_weight"])

    return realistic_delta, first_pass


# ============================================================
# RESULTING-ARSENAL-SHAPE GUARDRAIL
# ============================================================

# a candidate is REJECTED if it would leave the pitcher with fewer
# than this many real (>= floor) pitch types -- 2 is the minimum
# viable arsenal size. This is a GENERAL rule, not a DROP-specific
# special case: it naturally covers "don't recommend dropping a
# pitch for a 2-pitch pitcher" as one consequence (dropping 1 of 2
# leaves exactly 1, which fails this check), but also catches a
# USAGE_DECREASE large enough to functionally gut a pitch below the
# floor without being formally tagged as a DROP event, or a 3-pitch
# pitcher's drop leaving them with only 2 -- one consistent rule
# instead of several special-cased ones.
MIN_RESULTING_PITCH_TYPES = 2


def check_resulting_arsenal(target_arsenal_row, current_pitch_types, pitch_type, change_type, realistic_delta):
    """
    Estimates the pitcher's arsenal AFTER this candidate's change,
    holding every OTHER pitch type at its current known usage (a
    simplification -- doesn't model full proportional redistribution
    across the whole arsenal to keep usage summing to 100%, but is
    sufficient for the specific question this guardrail asks: would
    this leave too few real weapons).

    Returns (is_valid, resulting_pitch_count, resulting_usage_for_this_pitch).
    """
    resulting_types = set(current_pitch_types)

    if change_type == "ADD":
        current_usage_for_pt = 0.0
    else:
        current_usage_for_pt = target_arsenal_row.get(f"{pitch_type}_usage_pct", 0.0)
        if pd.isna(current_usage_for_pt):
            current_usage_for_pt = 0.0

    resulting_usage_for_pt = current_usage_for_pt + realistic_delta

    if resulting_usage_for_pt >= MEANINGFUL_USAGE_FLOOR:
        resulting_types.add(pitch_type)
    else:
        resulting_types.discard(pitch_type)

    is_valid = len(resulting_types) >= MIN_RESULTING_PITCH_TYPES

    return is_valid, len(resulting_types), resulting_usage_for_pt


# historical maximum usage_pct_after EVER observed for each pitch
# type, across all real change events -- used to flag (not exclude)
# a candidate whose resulting usage would exceed anything actually
# seen historically, a data-driven ceiling rather than an arbitrary
# picked percentage
HISTORICAL_MAX_USAGE_BY_PITCH_TYPE = training.groupby("pitch_type")["usage_pct_after"].max()


# ============================================================
# "NO CHANGE" BASELINE MODEL
# ============================================================

"""
Answers a real gap: every row recommend_arsenal_changes() produces
represents SOME change (ADD/DROP/INCREASE/DECREASE) -- there was no
explicit "if this pitcher changes nothing at all" row to compare
those options against. Without that, a top-ranked recommendation
can't be judged against the real alternative of standing pat.

DESIGN: a SEPARATE, GENERAL regression -- outcome(t+1) ~ outcome(t)
-- fit ONCE across ALL consecutive pitcher-seasons in the outcomes
data, NOT filtered to any specific arsenal-change type. This is
deliberately NOT similarity-weighted to a specific target the way
predict_arsenal_change_effect()'s candidates are: "how much does
performance persist year over year in general" is a property of the
outcome metric's own behavior across the whole population, not
something that should be borrowed from whichever pitch type's
comparison group a specific candidate happens to use. Weighted by
the same BIP-based reliability already used everywhere else in this
pipeline.

HONEST EXPECTATION: given Check 4 (39_validate_pitch_type_
sensitivity.py) already found mean-reversion dominates the change-
specific signal even in the best case, this baseline will likely
land CLOSE to most ranked candidates, not dramatically different --
that's not a flaw in this baseline, it's the same finding this
project already validated, now made an explicit, visible comparison
point instead of an implicit one.
"""

_no_change_models = {}  # cached per outcome_metric, fit once


def build_no_change_baseline_model(outcome_metric):
    if outcome_metric in _no_change_models:
        return _no_change_models[outcome_metric]

    outcomes_from = outcomes[["player_id", "season", outcome_metric, "bip"]].rename(
        columns={"season": "season_from", outcome_metric: "outcome_from", "bip": "bip_from"}
    )
    outcomes_to = outcomes[["player_id", "season", outcome_metric, "bip"]].rename(
        columns={"season": "season_to", outcome_metric: "outcome_to", "bip": "bip_to"}
    )
    outcomes_from["season_to"] = outcomes_from["season_from"] + 1

    paired = outcomes_from.merge(outcomes_to, on=["player_id", "season_to"], how="inner")
    paired = paired.dropna(subset=["outcome_from", "outcome_to", "bip_to"])

    weights = reliability_weight(paired["bip_to"].values)

    X = paired[["outcome_from"]].values
    y = paired["outcome_to"].values

    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)

    r2 = model.score(X, y, sample_weight=weights)

    print(
        f"\nBuilt no-change baseline model for {outcome_metric}: "
        f"{len(paired):,} consecutive pitcher-season pairs, "
        f"in-sample weighted R2={r2:.3f}"
    )

    _no_change_models[outcome_metric] = model
    return model


def predict_no_change(target_features, target_outcome_before, outcome_metric, target_player_id, target_season, carrier_pitch_types=None):
    """
    Similarity-weighted 'no change' baseline -- uses the SAME
    weighted-regression architecture as every other candidate,
    evaluated with usage_pct_delta held at 0 (distance/synergy left
    to their group medians), rather than a separate, simpler,
    globally-unweighted regression.

    WHY THIS REPLACED THE ORIGINAL: 70_no_change_bias_investigation
    .py found the original NO_CHANGE model was NOT a fair comparison
    point -- it regressed EVERY target toward the GLOBAL population
    mean, while every other candidate implicitly regresses toward a
    LOCAL, target-tier-specific mean via similarity-weighted comps.
    For an above-average pitcher, this meant the original NO_CHANGE
    systematically over-predicted regression (looked artificially
    worse), which is the real reason it won 0 of 75 real pitchers in
    69_batch_review.py's sweep -- ruled out as survivorship bias or
    a "changes correlate with improvement" confound (both tested
    directly and found NOT to explain it); the actual cause was this
    architecture mismatch. Confirmed directly: the SAME similarity-
    weighted model evaluated at delta=0 landed at 0.3051 for a real
    test case, far closer to the real ADD prediction (0.3058) than
    the OLD NO_CHANGE model's 0.3115.

    Averages across a FEW real, viable ADD-type carrier pitch types
    rather than one arbitrary choice, so the result isn't sensitive
    to which specific pitch type happens to be used to access the
    similarity-weighted architecture.
    """
    if carrier_pitch_types is None:
        add_counts = training[training["change_type"] == "ADD"]["pitch_type"].value_counts()
        carrier_pitch_types = add_counts[add_counts >= MIN_GROUP_SIZE_FOR_REGRESSION].index.tolist()[:3]

    predictions = []
    for pitch_type in carrier_pitch_types:
        zero_delta_features = dict(target_features)
        zero_delta_features["usage_pct_delta"] = 0.0
        result = predict_arsenal_change_effect(
            zero_delta_features, pitch_type, "ADD", target_outcome_before,
            outcome_metric, target_player_id, target_season
        )
        if result["predicted_outcome"] is not None:
            predictions.append(result["predicted_outcome"])

    if len(predictions) == 0:
        # fallback to the original simple model only if NO
        # similarity-weighted carrier works at all for this target
        model = build_no_change_baseline_model(outcome_metric)
        return model.predict([[target_outcome_before]])[0]

    return float(np.median(predictions))


# ============================================================
# THE RECOMMENDATION FUNCTION
# ============================================================

# ============================================================
# CONFIDENCE TIER -- based on real, empirically-derived boundaries
# ============================================================

# 65_confidence_calibration.py confirmed n_historical_events carries
# REAL, if modest, calibration signal on genuinely unseen walk-
# forward data: MAE dropped from 0.0266 (41-50 comps) to 0.0240
# (110+ comps), a clean, monotonic trend across every bin, though
# only ~10% relative reduction -- real, not dramatic. These tier
# boundaries are collapsed from that same real result (65's 4 bins:
# 41-50, 50-86, 87-110, 110-303), not arbitrary round numbers.
# Labels are deliberately low-key given the modest effect size --
# NOT "high/low confidence," which would overstate how much accuracy
# actually differs between tiers.
def confidence_tier(n_historical_events):
    if pd.isna(n_historical_events):
        return "n/a"
    elif n_historical_events < 50:
        return "Limited evidence"
    elif n_historical_events < 110:
        return "Moderate evidence"
    else:
        return "Strong evidence"


def recommend_arsenal_changes(target_player_id, target_season, outcome_metric="xwoba_against", top_n=5, add_prediction_fn=None, other_change_prediction_fn=None, swap_prediction_fn=None, top_swap_candidates=2):
    """
    Given a target pitcher, searches across every viable candidate
    pitch type they don't already throw, predicts the effect of each,
    and returns a RANKED DataFrame of recommendations.

    add_prediction_fn: optional override for ADD-candidate
    predictions specifically. Defaults to the standard predict_
    arsenal_change_effect if not provided.

    other_change_prediction_fn: optional override for DROP/USAGE_
    INCREASE/USAGE_DECREASE candidates. Defaults to the standard
    predict_arsenal_change_effect if not provided -- existing
    behavior (including any caller that only passes add_prediction_
    fn, like 57_recommendation_engine_with_stage1.py) is completely
    unchanged unless this new parameter is also explicitly passed.

    Both exist so a higher-level script can inject predict_arsenal_
    change_with_stage1 (54) or predict_arsenal_change_with_stage1_
    all_types (60) without 45 needing to import from either
    directly, which would create a circular import (both already
    import FROM 45, via 50).

    swap_prediction_fn: optional -- e.g. 96's predict_swap_effect.
    If provided, generates real "SWAP" candidates (dropping one
    pitch and adding another together, as a single decision) from
    the top_swap_candidates viable DROP and ADD results already
    computed below, rather than the caller having to infer a swap
    by summing two independent DROP/ADD predictions -- confirmed
    directly that summing independently can badly overstate the
    real effect (both independent predictions include the same
    mean-reversion term, effectively double-counting it). Defaults
    to None -- no SWAP candidates, existing behavior is completely
    unchanged unless this is explicitly passed.
    """
    if add_prediction_fn is None:
        add_prediction_fn = predict_arsenal_change_effect
    if other_change_prediction_fn is None:
        other_change_prediction_fn = predict_arsenal_change_effect

    # look up target's current raw arsenal composition, to determine
    # which pitch types are real ADD candidates (not already thrown)
    target_key = (target_player_id, target_season)
    if target_key not in arsenal_raw_indexed.index:
        raise ValueError(
            f"No arsenal data found for player_id={target_player_id}, "
            f"season={target_season}."
        )
    target_arsenal_row = arsenal_raw_indexed.loc[target_key]
    if isinstance(target_arsenal_row, pd.DataFrame):
        target_arsenal_row = target_arsenal_row.iloc[0]

    current_pitch_types = set(get_qualifying_pitch_types(target_arsenal_row))

    # look up target's standardized similarity profile
    target_sim_rows = similarity_features[
        (similarity_features["player_id"] == target_player_id)
        & (similarity_features["season"] == target_season)
    ]
    if len(target_sim_rows) == 0:
        raise ValueError(
            f"No similarity feature profile found for player_id="
            f"{target_player_id}, season={target_season} -- was this "
            f"pitcher-season dropped for incomplete data in "
            f"37_similarity_feature_space.py?"
        )
    target_sim_row = target_sim_rows.iloc[0]
    target_features = {c: target_sim_row[c] for c in SIMILARITY_FEATURE_COLS}
    # NOTE: usage_pct_delta is intentionally NOT set here anymore --
    # it's estimated per-candidate below (realistic, similarity-
    # weighted historical precedent for ADD/INCREASE/DECREASE; the
    # target's own exact known usage for DROP) rather than one flat
    # assumed number applied to every candidate alike

    target_name = target_sim_row.get("player_name", f"player_id {target_player_id}")

    # look up target's current baseline outcome (the lagged predictor)
    target_baseline_rows = outcomes[outcomes["player_id"] == target_player_id]
    if len(target_baseline_rows) == 0 or outcome_metric not in target_baseline_rows.columns:
        raise ValueError(
            f"No {outcome_metric} baseline found for player_id="
            f"{target_player_id} in the outcomes file."
        )
    # use the most recent available season as the baseline
    target_baseline_rows = target_baseline_rows.sort_values("season")
    target_outcome_before = target_baseline_rows[outcome_metric].iloc[-1]

    print(f"\nTarget: {target_name} (player_id={target_player_id}, season={target_season})")
    print(f"Current arsenal: {sorted(current_pitch_types)}")

    # NO_CHANGE baseline -- the real comparison point every other
    # candidate should be judged against. Now uses the SAME
    # similarity-weighted architecture as every other candidate
    # (see predict_no_change's docstring -- 70_no_change_bias_
    # investigation.py found the original, simpler version wasn't a
    # fair comparison point, which is why it never won across a real
    # 75-pitcher sweep)
    no_change_predicted = predict_no_change(target_features, target_outcome_before, outcome_metric, target_player_id, target_season)
    results = [{
        "change_type": "NO_CHANGE",
        "pitch_type": "(none)",
        "estimated_usage_delta": 0.0,
        "predicted_outcome": no_change_predicted,
        "predicted_change": no_change_predicted - target_outcome_before,
        "n_historical_events": np.nan,
        "confidence_tier": "n/a",
        "method": "no_change_baseline",
        "fallback_values_used": "none",
    }]

    # build the full candidate list across all four change types --
    # ADD candidates are pitch types NOT currently thrown; DROP/
    # USAGE_INCREASE/USAGE_DECREASE candidates are pitch types the
    # target ALREADY throws (you can't drop or re-weight a pitch you
    # don't have)
    candidates = []  # list of (change_type, pitch_type)

    for change_type in CHANGE_TYPES_TO_SEARCH:
        change_counts = training[training["change_type"] == change_type]["pitch_type"].value_counts()
        viable_for_this_change = set(change_counts[change_counts >= MIN_GROUP_SIZE_FOR_REGRESSION].index.tolist())

        if change_type == "ADD":
            eligible_types = viable_for_this_change - current_pitch_types
        else:
            eligible_types = viable_for_this_change & current_pitch_types

        for pt in sorted(eligible_types):
            candidates.append((change_type, pt))

    print(f"Candidates found across all change types (viable precedent + eligible): {candidates}")

    if len(candidates) == 0:
        print("No viable candidates found for this pitcher -- returning NO_CHANGE baseline only.")
        results_df = pd.DataFrame(results)
        results_df.insert(0, "rank", 1)
        return results_df, {}

    result_objects = {}

    for change_type, pitch_type in candidates:
        # DROP has an EXACT, already-known delta (current usage -> 0)
        # -- no estimation needed, unlike ADD/INCREASE/DECREASE where
        # the post-change usage level is genuinely unknown in advance
        if change_type == "DROP":
            current_usage = target_arsenal_row.get(f"{pitch_type}_usage_pct", np.nan)
            if pd.isna(current_usage):
                continue
            known_delta = -current_usage
        else:
            known_delta = None

        realistic_delta, _ = estimate_realistic_usage_delta(
            target_features, pitch_type, change_type, target_player_id,
            target_season, target_outcome_before, outcome_metric,
            known_delta=known_delta
        )

        if realistic_delta is None:
            continue

        # GUARDRAIL: exclude candidates that would leave the pitcher
        # with too few real weapons -- see MIN_RESULTING_PITCH_TYPES
        # above for why this is a general rule, not just a DROP-
        # specific special case
        is_valid, resulting_pitch_count, resulting_usage = check_resulting_arsenal(
            target_arsenal_row, current_pitch_types, pitch_type, change_type, realistic_delta
        )
        if not is_valid:
            print(
                f"  excluded: {change_type} {pitch_type} would leave only "
                f"{resulting_pitch_count} real pitch type(s) (minimum is "
                f"{MIN_RESULTING_PITCH_TYPES})"
            )
            continue

        # GUARDRAIL: never recommend fully dropping the pitcher's own
        # real anchor pitch (their primary fastball-type pitch --
        # anchor_type is always FF or SI, real data covers ~99% of
        # pitcher-seasons). This is a domain-knowledge exclusion, NOT
        # a statistically-derived one: direct testing (usage magnitude
        # and usage-beyond-historical-ceiling, both checked against
        # real subsequent outcomes) found no significant relationship
        # (p=0.76, p=0.42) to justify a data-driven penalty. The real
        # justification is structural -- this model evaluates each
        # pitch's own isolated quality and has no way to see a fastball's
        # tunneling/setup value for the rest of the arsenal, so a full
        # drop of the anchor pitch specifically is excluded on that
        # basis, disclosed here as exactly that, not dressed up as a
        # statistical finding.
        target_anchor_type = target_sim_row.get("anchor_type")
        if change_type == "DROP" and pd.notna(target_anchor_type) and pitch_type == target_anchor_type:
            print(
                f"  excluded: DROP {pitch_type} is this pitcher's own "
                f"anchor pitch (primary fastball-type pitch) -- excluded "
                f"by design, not by statistical evidence (see guardrail "
                f"comment in code)"
            )
            continue

        # FLAG (don't exclude) if the resulting usage would exceed
        # anything ever actually observed historically for this
        # specific pitch type -- a data-driven ceiling, not an
        # excluding rule, since "unprecedented" isn't the same as
        # "impossible"
        historical_max = HISTORICAL_MAX_USAGE_BY_PITCH_TYPE.get(pitch_type, np.nan)
        exceeds_historical_precedent = (
            pd.notna(historical_max) and resulting_usage > historical_max
        )

        final_target_features = dict(target_features)
        final_target_features["usage_pct_delta"] = realistic_delta

        # ADD candidates use whichever function was injected (base by
        # default, or predict_arsenal_change_with_stage1 when a
        # higher-level caller provides it) -- DROP/USAGE_INCREASE/
        # USAGE_DECREASE always use the base function, since Stage 1
        # only applies to newly-added pitches
        prediction_fn = add_prediction_fn if change_type == "ADD" else other_change_prediction_fn

        result = prediction_fn(
            final_target_features, pitch_type, change_type,
            target_outcome_before, outcome_metric,
            target_player_id, target_season
        )

        if result["predicted_outcome"] is None:
            continue

        # disclose explicitly when distance/synergy fell back to a
        # historical median rather than a known value -- a real,
        # honest consequence of this being a PROSPECTIVE
        # recommendation for a pitch the target hasn't thrown yet
        used_fallback_values = []
        if result.get("used_distance_feature") and "new_pitch_nn_distance_from_existing" not in final_target_features:
            used_fallback_values.append("distance (historical median)")
        if result.get("used_synergy_feature") and "arsenal_synergy_whiff_delta" not in final_target_features:
            used_fallback_values.append("synergy (historical median)")

        results.append({
            "change_type": change_type,
            "pitch_type": pitch_type,
            "estimated_usage_delta": realistic_delta,
            "resulting_usage": resulting_usage,
            "exceeds_historical_precedent": exceeds_historical_precedent,
            "predicted_outcome": result["predicted_outcome"],
            "predicted_change": result["predicted_change"],
            "n_historical_events": result["n_historical_events"],
            "confidence_tier": confidence_tier(result["n_historical_events"]),
            "method": result["method"],
            "fallback_values_used": ", ".join(used_fallback_values) if used_fallback_values else "none",
        })
        result_objects[(change_type, pitch_type)] = result

    # SWAP candidates -- real, combined predictions for dropping one
    # pitch and adding another together, generated from the top real
    # DROP and ADD candidates already found above. Only built if a
    # swap_prediction_fn was explicitly provided (see docstring) --
    # existing behavior is unchanged otherwise.
    if swap_prediction_fn is not None and len(results) > 0:
        interim_df = pd.DataFrame(results)
        top_drops = (
            interim_df[interim_df["change_type"] == "DROP"]
            .sort_values("predicted_change", ascending=(outcome_metric in LOWER_IS_BETTER_METRICS))
            .head(top_swap_candidates)["pitch_type"].tolist()
        )
        top_adds = (
            interim_df[interim_df["change_type"] == "ADD"]
            .sort_values("predicted_change", ascending=(outcome_metric in LOWER_IS_BETTER_METRICS))
            .head(top_swap_candidates)["pitch_type"].tolist()
        )

        for drop_pt in top_drops:
            for add_pt in top_adds:
                swap_result = swap_prediction_fn(target_player_id, target_season, drop_pt, add_pt)
                if swap_result.get("predicted_change") is None:
                    continue
                swap_pitch_label = f"{drop_pt} -> {add_pt}"
                results.append({
                    "change_type": "SWAP",
                    "pitch_type": swap_pitch_label,
                    "estimated_usage_delta": np.nan,
                    "resulting_usage": np.nan,
                    "exceeds_historical_precedent": False,
                    "predicted_outcome": swap_result["predicted_outcome"],
                    "predicted_change": swap_result["predicted_change"],
                    "n_historical_events": swap_result["n_real_events_used"],
                    "confidence_tier": confidence_tier(swap_result["n_real_events_used"]),
                    "method": "swap_similarity_weighted_regression",
                    "fallback_values_used": "none",
                })
                # store the real comp group the same way every other
                # candidate type does, so downstream code (94's batch
                # precompute, show_similar_comps) can look up real
                # swap comps the same way it already does for
                # everything else
                if "group" in swap_result:
                    result_objects[("SWAP", swap_pitch_label)] = {"group": swap_result["group"]}

    if len(results) == 0:
        print("No candidates produced a usable prediction.")
        return pd.DataFrame(), {}

    results_df = pd.DataFrame(results)

    # rank by predicted improvement -- direction depends on the metric
    ascending = outcome_metric in LOWER_IS_BETTER_METRICS
    results_df = results_df.sort_values("predicted_change", ascending=ascending).reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    return results_df, result_objects


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("\n\n==============================")
    print("DEMONSTRATION")
    print("==============================")

    # real target pitcher from the actual similarity features file --
    # picking a mid-list, non-trivial profile rather than row 0 again
    demo_row = similarity_features.dropna(subset=SIMILARITY_FEATURE_COLS).iloc[
        len(similarity_features) // 3
    ]
    demo_player_id = demo_row["player_id"]
    demo_season = demo_row["season"]

    recommendations, result_objects = recommend_arsenal_changes(
        demo_player_id, demo_season, outcome_metric="xwoba_against", top_n=5
    )

    if len(recommendations) > 0:
        print("\n\n==============================")
        print("Ranked recommendations")
        print("==============================")
        print(
            "NOTE: fallback_values_used shows when a candidate's distance/\n"
            "synergy features were unknown (the target hasn't actually\n"
            "thrown this pitch yet) and defaulted to that comparison\n"
            "group's historical median -- a real, disclosed assumption,\n"
            "not perfect foreknowledge.\n"
        )
        print(recommendations.to_string(index=False))

        top_row = recommendations.iloc[0]
        top_key = (top_row["change_type"], top_row["pitch_type"])
        print(
            f"\n\nTop recommendation: {top_row['change_type']} {top_row['pitch_type']} "
            f"(estimated usage delta: {top_row['estimated_usage_delta']:+.1f}pp)"
        )
        print(f"Most similar historical comps who made this change:")
        show_similar_comps(result_objects[top_key])
