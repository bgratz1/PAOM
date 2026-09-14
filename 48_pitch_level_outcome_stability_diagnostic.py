"""
48_pitch_level_outcome_stability_diagnostic.py

Purpose:
--------
47_pull_pitch_level_outcomes.py's real run confirmed genuine pitch-
type-level data (median 4 rows per pitcher-season), but also showed
xwoba_against ranging from 0.000 to 2.017 -- neither is a plausible
value for a real, meaningfully-used pitch, and both are the
signature of small-sample instability. Splitting a season's worth of
outcomes down to the pitch-type level inherently trades sample size
for granularity, so this instability is EXPECTED, not a surprise --
but it needs the same treatment the season-wide FIP/xwOBA data
already went through (33_outcome_stability_diagnostic.py), likely
more aggressively given the smaller samples involved here.

Same three checks as 33, adapted to this new dataset's actual
columns -- bucketing by "pitches" (the pitch-type-level sample size)
instead of "bip" (the season-wide signal 33 used), and checking all
three outcome columns this pull returned (xwoba_against, whiff_pct,
run_value_per_100), not just one:

1. Bucket pitcher-pitch-type rows by pitch count, show each bucket's
   outcome spread -- direct evidence of where instability actually
   lives, not just a guess from the two extreme values already seen.
2. List the most extreme outcome values league-wide, alongside their
   pitch count -- see how many "0.000/2.017-style" cases exist and
   whether they cluster at low pitch counts.
3. Sweep candidate MIN_PITCHES thresholds, reporting rows/pitchers
   retained (coverage cost) against resulting spread (stability
   benefit) -- same coverage-vs-stability tradeoff used throughout
   this project's cutoff work, applied here to pick an evidence-
   based threshold instead of another guess.

Output:
-------
Printed diagnostics only -- no file saved. Use the results to decide
a MIN_PITCHES floor (or confidence-shrinkage approach) before this
data feeds into anything downstream.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "pitcher_pitch_level_outcomes_2020_2025.csv"

# pitch-type-level counts run smaller on average than season-total
# BIP (one pitch type is a SUBSET of a pitcher's whole workload) --
# bucket edges start much lower than 33's did
PITCHES_BUCKETS = [1, 10, 25, 50, 100, 200, 400, 5000]
CANDIDATE_MIN_PITCHES = [1, 10, 25, 50, 75, 100, 150, 200, 300]

N_EXTREME_TO_SHOW = 15

METRICS_TO_CHECK = ["xwoba_against", "whiff_pct", "run_value_per_100"]


# ============================================================
# LOAD
# ============================================================

print("Loading pitch-level outcome data...")

df = pd.read_csv(INPUT_FILE)

required_cols = ["player_id", "season", "pitch_type", "pitches"] + METRICS_TO_CHECK
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}. This script needs "
        f"47_pull_pitch_level_outcomes.py's output."
    )

print(f"Loaded {len(df):,} pitcher-pitch-type-season rows\n")


# ============================================================
# CHECK 1: OUTCOME SPREAD BY PITCH-COUNT BUCKET
# ============================================================

print("==============================")
print("CHECK 1: Outcome spread by pitch-count bucket")
print("==============================")
print(
    "If low-pitch-count buckets show dramatically wider spread (std,\n"
    "min, max) than high-count buckets, that's direct evidence of\n"
    "small-sample instability, not just the two extreme values\n"
    "already seen in 47's own output.\n"
)

df["pitches_bucket"] = pd.cut(df["pitches"], bins=PITCHES_BUCKETS, right=False)

for metric in METRICS_TO_CHECK:
    print(f"\n--- {metric} ---")
    bucket_stats = df.groupby("pitches_bucket", observed=True)[metric].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(3)
    print(bucket_stats)


# ============================================================
# CHECK 2: MOST EXTREME OUTCOME VALUES, WITH THEIR PITCH COUNT
# ============================================================

print("\n\n==============================")
print("CHECK 2: Most extreme outcome values (top and bottom), with pitch count")
print("==============================")

for metric in METRICS_TO_CHECK:
    print(f"\n--- {metric}: highest values ---")
    top = df.nlargest(N_EXTREME_TO_SHOW, metric)[["player_id", "season", "pitch_type", "pitches", metric]]
    print(top.to_string(index=False))

    print(f"\n--- {metric}: lowest values ---")
    bottom = df.nsmallest(N_EXTREME_TO_SHOW, metric)[["player_id", "season", "pitch_type", "pitches", metric]]
    print(bottom.to_string(index=False))

    extreme_combined = pd.concat([top, bottom])
    median_pitches = df["pitches"].median()
    extreme_median_pitches = extreme_combined["pitches"].median()
    print(
        f"\nMedian pitches among these {metric} extremes: {extreme_median_pitches:.0f} "
        f"(population median pitches: {median_pitches:.0f}) -- "
        f"{'extremes skew toward LOW pitch counts, consistent with instability' if extreme_median_pitches < median_pitches else 'extremes do NOT obviously skew toward low pitch counts'}"
    )


# ============================================================
# CHECK 3: SWEEP CANDIDATE MIN_PITCHES THRESHOLDS
# ============================================================

print("\n\n==============================")
print("CHECK 3: Candidate MIN_PITCHES thresholds -- coverage vs. stability")
print("==============================")

results = []
for threshold in CANDIDATE_MIN_PITCHES:
    subset = df[df["pitches"] >= threshold]
    row = {
        "min_pitches": threshold,
        "rows_retained": len(subset),
        "pitchers_retained": subset["player_id"].nunique(),
        "pct_of_rows_retained": len(subset) / len(df),
    }
    for metric in METRICS_TO_CHECK:
        row[f"{metric}_std"] = subset[metric].std()
        row[f"{metric}_max_abs_from_median"] = (
            (subset[metric] - subset[metric].median()).abs().max()
        )
    results.append(row)

results_df = pd.DataFrame(results)
print(results_df.round(3).to_string(index=False))

print(
    "\nLook for where the *_std and *_max_abs_from_median columns "
    "stop dropping meaningfully as min_pitches increases -- that's "
    "roughly where real instability ends and you're just trading "
    "away coverage for no further stability benefit. rows_retained/ "
    "pct_of_rows_retained show the coverage cost of each candidate."
)
