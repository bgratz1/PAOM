"""
93_paom_direct_validation.py

Purpose:
--------
Direct validation of PAOM's recommendations against real outcomes --
NOT a comparison against the historical engine's own predictions
(that's 90/91/92), but an independent test of PAOM itself, using
the same real historical event data everything else in this project
has been validated against.

METHOD: for each year's PAOM recommendation (per change type), check
whether a REAL pitcher actually made that EXACT recommended change
the following season -- a real, matching event already sitting in
this project's training data, with a real, known outcome. Compare
the real average outcome for "PAOM recommended it, and the pitcher
actually did it" against the real average outcome for arsenal
changes of that type in general (the baseline). This tests PAOM
directly against reality, independent of the historical engine's
own model.

HONEST LIMITATION, upfront: the sample here will likely be small --
not every real pitcher happens to follow PAOM's specific top
recommendation. Pooled across all six years to maximize it; results
should be read with that caveat in mind.

Once this exists, a genuine numerical blend becomes possible: fit a
real regression using both the historical engine's own prediction
and a PAOM-endorsement signal against real outcomes (same style as
85), letting the actual coefficients determine PAOM's honest weight
rather than guessing at one.

Output:
-------
Printed real outcome comparison, PAOM-endorsed-and-followed vs.
baseline, per change type. No file saved -- investigation, not a
deliverable.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np
import pybaseball

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
OUTCOME_METRIC = "xwoba_against"


# ============================================================
# IMPORT REAL TRAINING DATA
# ============================================================

print("Loading real historical training data from 60...")

spec = importlib.util.spec_from_file_location(
    "stage1_all_change_types", "60_stage1_all_change_types.py"
)
stage60 = importlib.util.module_from_spec(spec)
sys.modules["stage1_all_change_types"] = stage60
spec.loader.exec_module(stage60)

training = stage60.training


# ============================================================
# CROSSWALK
# ============================================================

print("\nBuilding real player_id <-> PAOM-style name crosswalk...")

chadwick = pybaseball.chadwick_register().dropna(subset=["key_mlbam"])
chadwick["key_mlbam"] = chadwick["key_mlbam"].astype(int)
chadwick["paom_style_name"] = chadwick["name_last"] + ", " + chadwick["name_first"]
paom_name_to_id = dict(zip(chadwick["paom_style_name"], chadwick["key_mlbam"]))

print(f"Crosswalk built for {len(paom_name_to_id):,} real MLBAM player_ids")


# ============================================================
# BUILD REAL EVENT LOOKUP -- (player_id, season_from, pitch_type,
# change_type) -> real outcome_change, for fast matching
# ============================================================

events = training.dropna(subset=[f"{OUTCOME_METRIC}_before", f"{OUTCOME_METRIC}_after"]).copy()
events["outcome_change"] = events[f"{OUTCOME_METRIC}_after"] - events[f"{OUTCOME_METRIC}_before"]

event_lookup = {}
for _, row in events.iterrows():
    key = (row["player_id"], row["season_from"], row["pitch_type"], row["change_type"])
    event_lookup[key] = row["outcome_change"]

print(f"\n{len(event_lookup):,} real historical events indexed for matching")


# ============================================================
# BASELINE: average real outcome change, by change type, ALL events
# ============================================================

baseline_by_type = events.groupby("change_type")["outcome_change"].agg(["mean", "count"])
print(f"\nBaseline (ALL real events, by change type):")
print(baseline_by_type)


# ============================================================
# MATCH PAOM RECOMMENDATIONS TO REAL, ACTUALLY-FOLLOWED EVENTS
# ============================================================

def validate_paom_type(change_type, file_pattern, pick_col, filter_fn=None):
    print(f"\n\n==============================")
    print(f"Direct validation: {change_type}")
    print(f"==============================")

    matched_outcomes = []

    for year in YEARS:
        try:
            paom_df = pd.read_csv(file_pattern.format(year=year))
        except FileNotFoundError:
            continue

        if filter_fn is not None:
            paom_df = filter_fn(paom_df)

        for _, paom_row in paom_df.iterrows():
            player_id = paom_name_to_id.get(paom_row["player_name"])
            if player_id is None:
                continue

            pick = paom_row[pick_col]
            key = (player_id, year, pick, change_type)

            if key in event_lookup:
                matched_outcomes.append(event_lookup[key])

    n_matched = len(matched_outcomes)
    print(f"\n{n_matched} real, matching events found (PAOM recommended it, "
          f"and the pitcher actually made that exact change the following season)")

    if n_matched < 5:
        print("Too few matches for a meaningful comparison -- this is itself "
              "useful information: PAOM's specific recommendations rarely "
              "correspond to what real pitchers actually go on to do.")
        return

    paom_mean = np.mean(matched_outcomes)
    baseline_mean = baseline_by_type.loc[change_type, "mean"]
    baseline_n = baseline_by_type.loc[change_type, "count"]

    print(f"\nPAOM-endorsed-and-followed mean real outcome change: {paom_mean:.4f} (n={n_matched})")
    print(f"All real {change_type} events mean outcome change: {baseline_mean:.4f} (n={baseline_n:.0f})")
    print(f"Difference: {paom_mean - baseline_mean:+.4f} (negative = PAOM's endorsed changes "
          f"performed BETTER, since lower xwoba_against is better)")

    if paom_mean < baseline_mean - 0.005:
        print(f"\nVerdict: PAOM-endorsed changes show a REAL, meaningfully BETTER "
              f"real outcome than the baseline -- genuine, independent evidence "
              f"PAOM's own logic carries real value.")
    elif paom_mean < baseline_mean:
        print(f"\nVerdict: PAOM-endorsed changes show a SMALL edge over baseline -- "
              f"modest, not dramatic, real signal.")
    else:
        print(f"\nVerdict: PAOM-endorsed changes do NOT outperform the baseline -- "
              f"no independent evidence of real value from this test.")


# ============================================================
# RUN FOR ALL FOUR CHANGE TYPES
# ============================================================

validate_paom_type(
    "ADD", "PAOM_add_recommendations_{year}.csv", "candidate_pitch_type",
    filter_fn=lambda df: df[df["rank_within_pitcher"] == 1]
)

def drop_filter(df):
    df = df.copy()
    df["drop_score"] = (1 - df["effectiveness_pctl"]) + (1 - df["redundancy_pctl"])
    df["drop_rank"] = df.groupby("player_name")["drop_score"].rank(ascending=False, method="first")
    return df[(df["drop_candidate"] == True) & (df["drop_rank"] == 1)]

validate_paom_type(
    "DROP", "PAOM_drop_recommendations_{year}.csv", "pitch_type",
    filter_fn=drop_filter
)

def usage_increase_filter(df):
    df = df[df["recommendation"] == "increase usage"].copy()
    df["usage_rank"] = df.groupby("player_name")["pctl_gap"].rank(ascending=False, method="first")
    return df[df["usage_rank"] == 1]

validate_paom_type(
    "USAGE_INCREASE", "PAOM_usage_recommendations_{year}.csv", "pitch_type",
    filter_fn=usage_increase_filter
)

def usage_decrease_filter(df):
    df = df[df["recommendation"] == "decrease usage"].copy()
    df["usage_rank"] = df.groupby("player_name")["pctl_gap"].rank(ascending=True, method="first")
    return df[df["usage_rank"] == 1]

validate_paom_type(
    "USAGE_DECREASE", "PAOM_usage_recommendations_{year}.csv", "pitch_type",
    filter_fn=usage_decrease_filter
)
