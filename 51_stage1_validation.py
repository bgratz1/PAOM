"""
51_stage1_validation.py

Purpose:
--------
50_stage1_pitch_quality_model.py's demonstration produced plausible-
LOOKING numbers for a hypothetical case -- this is the actual test
of whether Stage 1's predictions track REAL outcomes, not just
whether they look reasonable. Every other feature in this project
(distance, synergy) got this same treatment before being trusted.

METHOD: a proper LEAVE-ONE-OUT validation across a real sample of
historical ADD events that have KNOWN ground truth (a real pitch-
level xwoba_against for that specific player_id + season_to +
pitch_type, from 47_pull_pitch_level_outcomes.py). For each sampled
event, the event's OWN row is excluded from its own comparison
group (via predict_new_pitch_quality's training_df override) before
predicting it -- otherwise the "prediction" could trivially include
its own answer.

Then, same permutation-test rigor already applied to the synergy
feature (43_synergy_placebo_test.py): a PLACEBO check shuffles the
REAL outcomes across the same sample and recomputes "correlation"
against the SAME predictions. If the real correlation is
meaningfully higher than the shuffled one, that's genuine evidence
Stage 1 is picking up real signal, not an artifact.

SAMPLE SIZE: a stratified sample (across pitch types, matching the
sampling approach already used in 39/41), not every single ADD
event -- each validation case requires a full predict_arsenal_
change_effect call, so this is scoped to stay practical rather than
exhaustive.

Output:
-------
Printed correlation, mean absolute error, and the real-vs-placebo
comparison. No file saved -- this is a validation check, not a
deliverable.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 50 (which cascades through 45 and 38)
# ============================================================

print("Loading training data and Stage 1 prediction function from 50...")

spec = importlib.util.spec_from_file_location(
    "stage1_pitch_quality_model", "50_stage1_pitch_quality_model.py"
)
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1_pitch_quality_model"] = stage1
spec.loader.exec_module(stage1)

training = stage1.training
pitch_level_indexed = stage1.pitch_level_indexed
predict_new_pitch_quality = stage1.predict_new_pitch_quality
PITCH_QUALITY_METRIC = stage1.PITCH_QUALITY_METRIC


# ============================================================
# SETTINGS
# ============================================================

N_SAMPLE_PER_PITCH_TYPE = 8  # stratified -- same sampling
                               # philosophy as 39/41's diverse-
                               # target approach, kept practical
                               # given each case needs a full
                               # predict_arsenal_change_effect call

RANDOM_SEED = 42


# ============================================================
# IDENTIFY REAL ADD EVENTS WITH KNOWN GROUND TRUTH
# ============================================================

print("\nIdentifying ADD events with known real pitch-level outcomes...")

add_events = training[training["change_type"] == "ADD"].copy()

def has_ground_truth(row):
    key = (row["player_id"], row["season_to"], row["pitch_type"])
    return key in pitch_level_indexed.index

add_events["has_ground_truth"] = add_events.apply(has_ground_truth, axis=1)
validatable = add_events[add_events["has_ground_truth"]].copy()

print(f"{len(validatable):,} of {len(add_events):,} real ADD events have known ground truth")

if len(validatable) == 0:
    raise ValueError("No ADD events have ground-truth pitch-level data -- nothing to validate.")

# stratified sample across pitch types
rng = np.random.default_rng(RANDOM_SEED)
sample_frames = []
for pt, group in validatable.groupby("pitch_type"):
    n = min(N_SAMPLE_PER_PITCH_TYPE, len(group))
    sample_frames.append(group.sample(n=n, random_state=RANDOM_SEED))
sample = pd.concat(sample_frames, ignore_index=True)

print(f"Sampled {len(sample):,} events across {sample['pitch_type'].nunique()} pitch types for validation")


# ============================================================
# LEAVE-ONE-OUT PREDICTION FOR EACH SAMPLED EVENT
# ============================================================

print("\nRunning leave-one-out predictions...")

results = []

for i, event in sample.iterrows():
    player_id = event["player_id"]
    season_from = event["season_from"]
    season_to = event["season_to"]
    pitch_type = event["pitch_type"]

    actual = pitch_level_indexed.get((player_id, season_to, pitch_type), np.nan)
    if pd.isna(actual):
        continue

    # LEAVE-ONE-OUT: exclude this event's own row from its own
    # comparison group -- otherwise the prediction could trivially
    # include its own answer
    loo_training = training[
        ~(
            (training["player_id"] == player_id)
            & (training["season_from"] == season_from)
            & (training["season_to"] == season_to)
            & (training["pitch_type"] == pitch_type)
        )
    ]

    result = predict_new_pitch_quality(
        player_id, season_from, pitch_type, training_df=loo_training
    )

    if result["predicted_pitch_quality"] is not None:
        results.append({
            "player_id": player_id,
            "season_to": season_to,
            "pitch_type": pitch_type,
            "predicted": result["predicted_pitch_quality"],
            "actual": actual,
            "n_comps": result.get("n_comps_with_pitch_level_data", np.nan),
        })

results_df = pd.DataFrame(results)

print(f"\n{len(results_df):,} of {len(sample):,} sampled events produced a usable leave-one-out prediction")


# ============================================================
# REAL CORRELATION
# ============================================================

print("\n\n==============================")
print("Stage 1 validation: predicted vs. actual pitch-level xwoba_against")
print("==============================")

if len(results_df) < 5:
    raise ValueError(
        f"Only {len(results_df)} usable predictions -- too few for a "
        f"meaningful correlation check."
    )

real_corr = results_df["predicted"].corr(results_df["actual"])
real_mae = (results_df["predicted"] - results_df["actual"]).abs().mean()

print(f"\nReal correlation (predicted vs. actual): {real_corr:.3f}")
print(f"Mean absolute error: {real_mae:.4f}")

print(f"\nSample of individual predictions:")
print(results_df[["player_id", "season_to", "pitch_type", "predicted", "actual", "n_comps"]].to_string(index=False))


# ============================================================
# PLACEBO CHECK -- SAME PERMUTATION-TEST RIGOR AS 43/44
# ============================================================

print("\n\n==============================")
print("Placebo check (shuffled actuals) -- same rigor as 43_synergy_placebo_test.py")
print("==============================")

rng2 = np.random.default_rng(RANDOM_SEED)
shuffled_actual = rng2.permutation(results_df["actual"].values)

placebo_corr = pd.Series(results_df["predicted"].values).corr(pd.Series(shuffled_actual))

print(f"\nReal correlation: {real_corr:.3f}")
print(f"Placebo (shuffled) correlation: {placebo_corr:.3f}")

if abs(real_corr) > abs(placebo_corr) + 0.15:
    print(
        "\nReal correlation is meaningfully higher than the shuffled "
        "placebo -- genuine evidence Stage 1 is picking up real "
        "signal, not an artifact of scale or regression mechanics."
    )
else:
    print(
        "\nReal correlation is NOT meaningfully higher than the "
        "shuffled placebo -- Stage 1's predictions may not be "
        "carrying real signal yet. Worth investigating further "
        "before trusting these predictions or feeding them into "
        "Stage 2."
    )
