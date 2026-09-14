"""
61_ranking_level_backtest.py

Purpose:
--------
58_season_wide_backtest.py tested POINT accuracy -- how close was
each individual prediction to reality. It never tested whether the
RANKING itself carries real information: does the model correctly
say "event A should outperform event B" when A really DID
outperform B? Point accuracy and ranking accuracy are different
questions -- a model could have modest point accuracy but still
rank things in roughly the right order, or vice versa.

METHOD: pairwise concordance (the same general idea behind a C-
statistic, a standard way to evaluate ranking quality). For every
PAIR of real historical ADD events, using ONLY pre-change
information (leave-one-out, so neither event's own outcome leaks
into its own prediction) -- does the model's PREDICTED ordering
(which event should show more improvement) match the REAL ordering
of what actually happened? 50% concordance = pure chance, the same
as a coin flip. Meaningfully above 50% is genuine evidence the
ranking carries real information, not just the individual numbers.

This directly answers a question point-accuracy tests can't: if you
ask this tool to rank two candidates against each other, does the
top-ranked one actually tend to be the better real choice?

Tested for BOTH the base and Stage-1-integrated approaches, so the
comparison shows whether Stage 1 improves RANKING quality, not just
point accuracy.

Output:
-------
Printed concordance rate for base and integrated approaches, with an
explicit verdict on whether either carries real ranking information.
No file saved -- this is a validation check, not a deliverable.
"""

import importlib.util
import sys
import itertools

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 54
# ============================================================

print("Loading base + integrated prediction functions from 54...")

spec = importlib.util.spec_from_file_location(
    "stage2_integration", "54_stage2_integration.py"
)
stage2 = importlib.util.module_from_spec(spec)
sys.modules["stage2_integration"] = stage2
spec.loader.exec_module(stage2)

training = stage2.training
similarity_features = stage2.similarity_features
predict_arsenal_change_effect = stage2.predict_arsenal_change_effect
predict_arsenal_change_with_stage1 = stage2.predict_arsenal_change_with_stage1
estimate_realistic_usage_delta = stage2.estimate_realistic_usage_delta
SIMILARITY_FEATURE_COLS = stage2.SIMILARITY_FEATURE_COLS


# ============================================================
# SETTINGS
# ============================================================

N_SAMPLE_PER_PITCH_TYPE = 8  # same as 58 -- keeps pairwise
                               # comparisons (~n*(n-1)/2) tractable
                               # while staying comparable to 58's
                               # established sample
OUTCOME_METRIC = "xwoba_against"
RANDOM_SEED = 42  # SAME seed as 58, same events sampled


# ============================================================
# SAMPLE REAL ADD EVENTS -- SAME sample as 58
# ============================================================

print("\nSampling real ADD events (same sample as 58)...")

add_events = training[training["change_type"] == "ADD"].copy()
add_events = add_events.dropna(subset=[f"{OUTCOME_METRIC}_after", f"{OUTCOME_METRIC}_before"])

sample_frames = []
for pt, group in add_events.groupby("pitch_type"):
    n = min(N_SAMPLE_PER_PITCH_TYPE, len(group))
    sample_frames.append(group.sample(n=n, random_state=RANDOM_SEED))
sample = pd.concat(sample_frames, ignore_index=True)

print(f"Sampled {len(sample):,} events across {sample['pitch_type'].nunique()} pitch types")


# ============================================================
# COMPUTE LEAVE-ONE-OUT PREDICTIONS FOR EACH EVENT, ONCE
# (pairs are formed afterward from these precomputed values --
# avoids redundant computation across n*(n-1)/2 pairs)
# ============================================================

def compute_predictions(predict_fn, is_integrated, label):
    print(f"\nComputing leave-one-out predictions ({label})...")
    rows = []
    for _, event in sample.iterrows():
        player_id = event["player_id"]
        season_from = event["season_from"]
        season_to = event["season_to"]
        pitch_type = event["pitch_type"]

        actual_change = event[f"{OUTCOME_METRIC}_after"] - event[f"{OUTCOME_METRIC}_before"]
        target_outcome_before = event[f"{OUTCOME_METRIC}_before"]

        target_sim_rows = similarity_features[
            (similarity_features["player_id"] == player_id)
            & (similarity_features["season"] == season_from)
        ]
        if len(target_sim_rows) == 0:
            continue
        target_features = {c: target_sim_rows.iloc[0][c] for c in SIMILARITY_FEATURE_COLS}

        loo_training = training[
            ~(
                (training["player_id"] == player_id)
                & (training["season_from"] == season_from)
                & (training["season_to"] == season_to)
                & (training["pitch_type"] == pitch_type)
            )
        ]

        realistic_delta, _ = estimate_realistic_usage_delta(
            target_features, pitch_type, "ADD", player_id,
            season_from, target_outcome_before, OUTCOME_METRIC
        )
        if realistic_delta is None:
            continue

        final_features = dict(target_features)
        final_features["usage_pct_delta"] = realistic_delta

        result = predict_fn(
            final_features, pitch_type, "ADD", target_outcome_before,
            OUTCOME_METRIC, player_id, season_from, training_df=loo_training
        )

        if is_integrated:
            success = result.get("stage1_integration_status") == "success"
        else:
            success = result["predicted_outcome"] is not None

        if success:
            rows.append({
                "event_id": f"{player_id}_{season_to}_{pitch_type}",
                "predicted_change": result["predicted_change"],
                "actual_change": actual_change,
            })

    return pd.DataFrame(rows)


base_predictions = compute_predictions(predict_arsenal_change_effect, False, "BASE")
integrated_predictions = compute_predictions(predict_arsenal_change_with_stage1, True, "INTEGRATED")

print(f"\n{len(base_predictions):,} usable BASE predictions")
print(f"{len(integrated_predictions):,} usable INTEGRATED predictions")


# ============================================================
# PAIRWISE CONCORDANCE
# ============================================================

def compute_concordance(df, label):
    if len(df) < 5:
        print(f"\n{label}: too few predictions ({len(df)}) for meaningful pairwise comparison")
        return None

    n_concordant = 0
    n_discordant = 0
    n_tied = 0

    for (i, row_a), (j, row_b) in itertools.combinations(df.iterrows(), 2):
        pred_diff = row_a["predicted_change"] - row_b["predicted_change"]
        actual_diff = row_a["actual_change"] - row_b["actual_change"]

        if pred_diff == 0 or actual_diff == 0:
            n_tied += 1
            continue

        if np.sign(pred_diff) == np.sign(actual_diff):
            n_concordant += 1
        else:
            n_discordant += 1

    total_comparable = n_concordant + n_discordant
    concordance_rate = n_concordant / total_comparable if total_comparable > 0 else np.nan

    print(f"\n{label} (n={len(df)} events, {total_comparable:,} comparable pairs):")
    print(f"  Concordant: {n_concordant:,}, Discordant: {n_discordant:,}, Tied: {n_tied:,}")
    print(f"  Concordance rate: {concordance_rate:.1%} (50% = pure chance)")

    return concordance_rate


print("\n\n==============================")
print("Pairwise concordance -- does the ranking carry real information?")
print("==============================")

base_concordance = compute_concordance(base_predictions, "BASE (no Stage 1)")
integrated_concordance = compute_concordance(integrated_predictions, "INTEGRATED (with Stage 1)")


# ============================================================
# VERDICT
# ============================================================

print("\n\n==============================")
print("Verdict")
print("==============================")

for label, rate in [("BASE", base_concordance), ("INTEGRATED", integrated_concordance)]:
    if rate is None:
        continue
    if rate > 0.55:
        print(
            f"\n{label}: {rate:.1%} concordance -- meaningfully above "
            f"chance (50%). The ranking DOES carry real information: "
            f"when this approach says event A should outperform event "
            f"B, it's right more often than not."
        )
    else:
        print(
            f"\n{label}: {rate:.1%} concordance -- close to chance "
            f"(50%). Not strong evidence the ranking itself carries "
            f"real information, even if individual point predictions "
            f"have some accuracy."
        )

if base_concordance is not None and integrated_concordance is not None:
    if integrated_concordance > base_concordance + 0.03:
        print(
            f"\nStage 1 integration improves RANKING quality too, not "
            f"just point accuracy ({base_concordance:.1%} -> "
            f"{integrated_concordance:.1%})."
        )
    elif integrated_concordance < base_concordance - 0.03:
        print(
            f"\nStage 1 integration does NOT clearly improve ranking "
            f"quality on this sample ({base_concordance:.1%} -> "
            f"{integrated_concordance:.1%}), despite improving point "
            f"accuracy in 58 -- worth noting these are genuinely "
            f"different questions."
        )
    else:
        print(
            f"\nStage 1 integration shows a similar ranking-quality "
            f"picture to BASE ({base_concordance:.1%} vs. "
            f"{integrated_concordance:.1%})."
        )
