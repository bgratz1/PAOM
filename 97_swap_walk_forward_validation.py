"""
97_swap_walk_forward_validation.py

Purpose:
--------
Validates the swap prediction model (96) against genuinely unseen
real data, using the SAME walk-forward methodology already
established for the single-change models (64) -- train on real
swap events with season_from <= 2023, test ONLY on real swap events
with season_from >= 2024 (the change itself happened in a season the
model never saw during training).

WHY THIS MATTERS: the single-change models are extensively validated
(placebo tests, walk-forward, ranking concordance). Before swap
candidates influence real dashboard recommendations, this applies
the same standard -- not a lower bar just because the sample is
smaller.

Real, honest sample-size note (checked before committing to this
design, per direct discussion): 72 real swap transitions have
season_from >= 2024, producing a real, usable (if smaller than
single-change validation) test set -- confirmed NOT "genuinely too
thin" before proceeding with walk-forward rather than falling back
to leave-one-out.

METHOD: for each real test-set swap event, predict its outcome using
ONLY train-set swap events (season_from <= 2023) as the comparison
pool -- zero leakage, the model never sees anything about a 2024+
event when predicting it. Compares against two baselines: (1) a
naive NO_CHANGE baseline (predict no change at all), and (2) a naive
"predict the training mean" baseline -- the two standards the rest
of this project's validation work has always checked against.

Output:
-------
Printed MAE comparison (model vs. both baselines) and ranking
concordance, on the genuinely held-out real test set.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 96
# ============================================================

print("Loading swap prediction machinery from 96...")

spec = importlib.util.spec_from_file_location("swap_prediction", "96_swap_prediction.py")
swap_mod = importlib.util.module_from_spec(spec)
sys.modules["swap_prediction"] = swap_mod
spec.loader.exec_module(swap_mod)

swap_events = swap_mod.swap_events
predict_swap_effect = swap_mod.predict_swap_effect
outcome_indexed = swap_mod.outcome_indexed


# ============================================================
# REAL TRAIN/TEST SPLIT -- season_from <= 2023 vs >= 2024
# ============================================================

CUTOFF_YEAR = 2023

train_events = swap_events[swap_events["season_from"] <= CUTOFF_YEAR].reset_index(drop=True)
test_events_raw = swap_events[swap_events["season_from"] > CUTOFF_YEAR].reset_index(drop=True)

# REAL INDEPENDENCE ISSUE, caught and fixed: a single real transition
# (one player_id, one season_from) can produce multiple swap-pair
# rows (a pitcher who dropped 2 pitches and added 1 contributes 2
# rows, both sharing the exact same real, combined outcome_change).
# Validating across all raw rows would treat non-independent
# duplicates as separate evidence, artificially inflating apparent
# accuracy. Deduplicated to ONE row per real transition -- the
# pair involving the pitch with the highest real pre-swap usage,
# since that's the most significant real change within a multi-
# change transition, not an arbitrary pick.
test_events = (
    test_events_raw.sort_values("dropped_usage_pct_before", ascending=False)
    .drop_duplicates(subset=["player_id", "season_from"], keep="first")
    .reset_index(drop=True)
)

print(f"\nTrain set (season_from <= {CUTOFF_YEAR}): {len(train_events):,} real swap-pair events")
print(f"Test set, RAW rows (season_from > {CUTOFF_YEAR}): {len(test_events_raw):,}")
print(f"Test set, DEDUPLICATED to one row per real transition: {len(test_events):,} "
      f"(this is what validation below actually uses)")

if len(test_events) < 15:
    print(
        "\nHONEST FLAG: test set is smaller than this project's usual "
        "minimum group size threshold (15) -- results below should be "
        "read with real caution about sample size, even though this "
        "was checked as viable before starting."
    )


# ============================================================
# RUN THE REAL WALK-FORWARD VALIDATION
# ============================================================

results = []

for _, test_row in test_events.iterrows():
    pred = predict_swap_effect(
        test_row["player_id"], test_row["season_from"],
        test_row["dropped_pitch_type"], test_row["added_pitch_type"],
        swap_events_df=train_events  # ONLY train-set events -- the
                                        # model never sees this real
                                        # test event or anything from
                                        # its own season_from
    )
    if pred.get("predicted_change") is None:
        continue

    results.append({
        "player_id": test_row["player_id"],
        "season_from": test_row["season_from"],
        "dropped_pitch_type": test_row["dropped_pitch_type"],
        "added_pitch_type": test_row["added_pitch_type"],
        "real_outcome_change": test_row["outcome_change"],
        "predicted_change": pred["predicted_change"],
        "n_real_events_used": pred["n_real_events_used"],
    })

results_df = pd.DataFrame(results)
print(f"\n{len(results_df):,} of {len(test_events):,} real test events produced a usable prediction "
      f"(rest lacked sufficient real train-set precedent)")

if len(results_df) == 0:
    print("\nNo usable predictions -- cannot validate further.")
    sys.exit()


# ============================================================
# MAE vs. TWO REAL BASELINES
# ============================================================

model_mae = (results_df["predicted_change"] - results_df["real_outcome_change"]).abs().mean()

# baseline 1: NO_CHANGE (predict zero change)
no_change_mae = results_df["real_outcome_change"].abs().mean()

# baseline 2: predict the TRAIN SET's own real mean outcome_change
# (a naive "average swap effect" baseline, using only train data --
# no leakage)
train_mean_change = train_events["outcome_change"].mean()
train_mean_mae = (results_df["real_outcome_change"] - train_mean_change).abs().mean()

print(f"\n\n==============================")
print(f"REAL WALK-FORWARD RESULTS (genuinely unseen 2024+ swap events)")
print(f"==============================")
print(f"\nModel MAE: {model_mae:.4f}")
print(f"NO_CHANGE baseline MAE: {no_change_mae:.4f}")
print(f"Train-mean baseline MAE: {train_mean_mae:.4f}")

improvement_vs_no_change = (no_change_mae - model_mae) / no_change_mae * 100
improvement_vs_train_mean = (train_mean_mae - model_mae) / train_mean_mae * 100
print(f"\nModel improvement over NO_CHANGE: {improvement_vs_no_change:+.1f}%")
print(f"Model improvement over train-mean baseline: {improvement_vs_train_mean:+.1f}%")


# ============================================================
# RANKING CONCORDANCE (if sample allows)
# ============================================================

print(f"\n\n==============================")
print(f"RANKING CONCORDANCE")
print(f"==============================")

if len(results_df) < 10:
    print(
        f"Only {len(results_df)} usable test predictions -- too few "
        f"for a meaningful pairwise concordance check (this project's "
        f"other concordance tests used hundreds of pairs minimum). "
        f"Skipped rather than reported as a misleadingly precise "
        f"number."
    )
else:
    correct = 0
    total = 0
    for i in range(len(results_df)):
        for j in range(i + 1, len(results_df)):
            real_i, real_j = results_df.iloc[i]["real_outcome_change"], results_df.iloc[j]["real_outcome_change"]
            pred_i, pred_j = results_df.iloc[i]["predicted_change"], results_df.iloc[j]["predicted_change"]
            if real_i == real_j:
                continue
            total += 1
            if (real_i < real_j) == (pred_i < pred_j):
                correct += 1

    if total > 0:
        concordance = correct / total
        print(f"Pairwise concordance: {concordance:.1%} ({correct} of {total} pairs, "
              f"50% = chance level)")
    else:
        print("No valid pairs to compare.")


# ============================================================
# HONEST SUMMARY
# ============================================================

print(f"\n\n==============================")
print(f"SUMMARY")
print(f"==============================")
if model_mae < no_change_mae and model_mae < train_mean_mae:
    print(
        "The swap model beats BOTH real baselines on genuinely unseen "
        "data -- real, if modest given sample size, evidence it's "
        "learning something beyond pure regression-to-the-mean."
    )
elif model_mae < no_change_mae:
    print(
        "The swap model beats the NO_CHANGE baseline but not the "
        "train-mean baseline -- weak evidence; may mostly be picking "
        "up on a general 'swaps tend to help' pattern rather than "
        "anything target-specific."
    )
else:
    print(
        "The swap model does NOT clearly beat naive baselines on this "
        "held-out test -- given the small sample, this could be "
        "sample noise rather than a real failure, but it should NOT "
        "be treated as validated evidence the model works."
    )


# ============================================================
# PLACEBO TEST -- is dropped_usage_pct_before real signal, matching
# this project's established placebo-testing discipline (43/44)?
# ============================================================

print(f"\n\n==============================")
print(f"PLACEBO TEST: dropped_usage_pct_before")
print(f"==============================")

np.random.seed(42)
placebo_train = train_events.copy()
placebo_train["dropped_usage_pct_before"] = np.random.permutation(
    placebo_train["dropped_usage_pct_before"].values
)

placebo_results = []
for _, test_row in test_events.iterrows():
    pred = predict_swap_effect(
        test_row["player_id"], test_row["season_from"],
        test_row["dropped_pitch_type"], test_row["added_pitch_type"],
        swap_events_df=placebo_train
    )
    if pred.get("predicted_change") is None:
        continue
    placebo_results.append({
        "real_outcome_change": test_row["outcome_change"],
        "predicted_change": pred["predicted_change"],
    })

placebo_df = pd.DataFrame(placebo_results)
if len(placebo_df) > 0:
    placebo_mae = (placebo_df["predicted_change"] - placebo_df["real_outcome_change"]).abs().mean()
    print(f"\nReal model MAE: {model_mae:.4f}")
    print(f"Placebo (shuffled usage magnitude) MAE: {placebo_mae:.4f}")
    if model_mae < placebo_mae:
        print(
            "\nReal version outperforms its own placebo -- genuine "
            "evidence dropped_usage_pct_before carries real signal, "
            "not just noise the model happens to fit."
        )
    else:
        print(
            "\nReal version does NOT clearly outperform its own "
            "placebo -- the usage-magnitude predictor's real "
            "contribution is not confirmed by this test."
        )
else:
    print("Placebo run produced no usable predictions -- cannot compare.")
