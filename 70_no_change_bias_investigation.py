"""
70_no_change_bias_investigation.py

Purpose:
--------
69_batch_review.py found NO_CHANGE ranked #1 in 0 of 75 real,
randomly-sampled pitchers -- suspiciously absolute for something
that should occasionally be the genuinely best real choice. This
tests the most plausible explanation directly: NO_CHANGE's model is
fit on ALL consecutive pitcher-season pairs, while every OTHER
candidate's model is fit only on pitchers who made a REAL,
qualifying arsenal change AND have a real, measurable follow-up
season. That's a real potential selection effect -- a pitcher who
got worse and lost their role is more likely to be MISSING from an
event-based comparison group (no qualifying follow-up season to
measure) than from the general population NO_CHANGE draws from.

DIRECT TEST: compare the actual outcome distribution of NO_CHANGE's
training population against the actual outcome distribution of the
event-based comparison-group population. If the event population
shows a systematically better average outcome, that's a real,
structural bias -- not a bug in any one function, but a genuine
mismatch in what each model's training population represents.

Output:
-------
Printed comparison of outcome distributions, both populations, plus
a direct verdict on whether this explains 69's finding.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 45
# ============================================================

print("Loading training data and models from 45...")

spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine", "45_similarity_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(spec)
sys.modules["similarity_recommendation_engine"] = rec_engine
spec.loader.exec_module(rec_engine)

training = rec_engine.training
outcomes = rec_engine.outcomes

OUTCOME_METRIC = "xwoba_against"


# ============================================================
# POPULATION 1: NO_CHANGE's training population (ALL consecutive
# pitcher-season pairs, same construction as build_no_change_
# baseline_model in 45)
# ============================================================

print("\nBuilding NO_CHANGE's training population (all consecutive pitcher-seasons)...")

outcomes_from = outcomes[["player_id", "season", OUTCOME_METRIC, "bip"]].rename(
    columns={"season": "season_from", OUTCOME_METRIC: "outcome_from", "bip": "bip_from"}
)
outcomes_to = outcomes[["player_id", "season", OUTCOME_METRIC, "bip"]].rename(
    columns={"season": "season_to", OUTCOME_METRIC: "outcome_to", "bip": "bip_to"}
)
outcomes_from["season_to"] = outcomes_from["season_from"] + 1
no_change_population = outcomes_from.merge(outcomes_to, on=["player_id", "season_to"], how="inner")
no_change_population = no_change_population.dropna(subset=["outcome_from", "outcome_to"])

print(f"NO_CHANGE population: {len(no_change_population):,} consecutive pitcher-season pairs")


# ============================================================
# POPULATION 2: event-based comparison-group population (real
# ADD/DROP/USAGE_INCREASE/USAGE_DECREASE events with known outcomes
# -- the SAME population every other candidate's model draws from)
# ============================================================

print("Building event-based population (real qualifying arsenal-change events)...")

event_population = training.dropna(subset=[f"{OUTCOME_METRIC}_before", f"{OUTCOME_METRIC}_after"]).copy()

print(f"Event population: {len(event_population):,} real arsenal-change events with known outcomes")


# ============================================================
# COMPARE OUTCOME DISTRIBUTIONS DIRECTLY
# ============================================================

print("\n\n==============================")
print("Direct comparison: does the event population show systematically BETTER outcomes?")
print("==============================")

# outcome_to for NO_CHANGE's population; outcome_after for the event
# population -- same underlying metric (xwoba_against), same
# "what happened THE FOLLOWING SEASON" concept
no_change_outcomes = no_change_population["outcome_to"]
event_outcomes = event_population[f"{OUTCOME_METRIC}_after"]

print(f"\nNO_CHANGE population outcome_to ({OUTCOME_METRIC}):")
print(no_change_outcomes.describe())

print(f"\nEvent population outcome_after ({OUTCOME_METRIC}):")
print(event_outcomes.describe())

no_change_mean = no_change_outcomes.mean()
event_mean = event_outcomes.mean()

print(f"\nNO_CHANGE population mean: {no_change_mean:.4f}")
print(f"Event population mean: {event_mean:.4f}")
print(f"Difference: {no_change_mean - event_mean:+.4f} (positive = event population has BETTER/lower xwoba_against)")

# ALSO compare the BEFORE-season distributions -- if the event
# population's BASELINE (pre-change) outcomes were ALREADY better
# on average, that's a DIFFERENT (also real) explanation: pitchers
# who make changes might already be BETTER pitchers on average, not
# just "surviving" into the follow-up season
no_change_before = no_change_population["outcome_from"]
event_before = event_population[f"{OUTCOME_METRIC}_before"]

print(f"\n\nBASELINE (pre-change/pre-period) comparison:")
print(f"NO_CHANGE population mean outcome_from: {no_change_before.mean():.4f}")
print(f"Event population mean outcome_before: {event_before.mean():.4f}")
print(f"Difference: {no_change_before.mean() - event_before.mean():+.4f}")


# ============================================================
# VERDICT
# ============================================================

print("\n\n==============================")
print("Verdict")
print("==============================")

after_diff = no_change_mean - event_mean
before_diff = no_change_before.mean() - event_before.mean()

if after_diff > 0.01:
    if abs(before_diff) < abs(after_diff) * 0.5:
        print(
            f"\nCONFIRMED, and it's a SURVIVORSHIP effect specifically: "
            f"the event population's follow-up-season outcomes are "
            f"{after_diff:.4f} BETTER on average than NO_CHANGE's "
            f"general population, but their BASELINE outcomes were "
            f"NOT meaningfully different ({before_diff:+.4f}) -- "
            f"pitchers who make a qualifying arsenal change AND have "
            f"real follow-up data available are a systematically "
            f"BETTER-OUTCOME subset than the general population, most "
            f"likely because worse-performing pitchers are more often "
            f"missing from event-based comparison groups (lost their "
            f"role, didn't throw enough the following year to "
            f"qualify) than from the general population. This is a "
            f"real, structural bias -- NOT a bug in any specific "
            f"function, but a genuine mismatch in what each model's "
            f"training population represents. It directly explains "
            f"why NO_CHANGE never wins: it's being compared against "
            f"models trained on a systematically rosier population."
        )
        print(
            "\nA REAL FIX: restrict NO_CHANGE's training population "
            "to the SAME kind of 'has a real, qualifying follow-up "
            "season' population the event-based models use, rather "
            "than the full unrestricted population -- an apples-to-"
            "apples comparison instead of comparing a broad baseline "
            "against a survivorship-biased one."
        )
    else:
        print(
            f"\nEvent population DOES show better average outcomes "
            f"({after_diff:+.4f}), but their BASELINE outcomes were "
            f"ALSO meaningfully better ({before_diff:+.4f}) -- this "
            f"looks more like pitchers who make changes tend to "
            f"already be better pitchers, not necessarily a pure "
            f"survivorship artifact. Still worth accounting for, but "
            f"a different mechanism than pure survivorship bias."
        )
else:
    print(
        f"\nNOT confirmed -- the two populations show similar average "
        f"outcomes ({after_diff:+.4f} difference). The 0% NO_CHANGE "
        f"win rate in 69 likely has a different explanation, worth "
        f"investigating further rather than attributing to this "
        f"population mismatch."
    )
