"""
68_run_value_stricter_floor.py

Purpose:
--------
48_pitch_level_outcome_stability_diagnostic.py already found run_
value_per_100 far noisier than xwoba_against or whiff_pct, even at
the MIN_PITCHES=50 floor applied to all three (max_abs_from_median
still 12.6 at that threshold, a genuinely large number for a "per
100 pitches" rate). The mechanism is understood: run value is built
from discrete, lumpy events (one home run swings it hugely), and the
"per 100 pitches" conversion AMPLIFIES that by rescaling a tiny raw
sample up to a 100-pitch basis -- a single home run at 1 pitch
becomes a wild, meaningless number once rescaled.

This has never actually been remediated -- disclosed, not fixed.
Before packaging this tool, that gap needs closing: sweeps
candidate floors ABOVE the existing 50-pitch cut (75/100/150/200/
250/300) specifically for run_value_per_100's OWN stabilization
point, separate from xwoba_against/whiff_pct (which were already
fine at 50 and don't need this).

NOTE ON THIS RUN'S DATA: the currently-saved pitcher_pitch_level_
outcomes_2020_2025.csv in this environment is a leftover synthetic
test file (16,095 rows), not the real one (13,357 rows) -- this run
confirms the MECHANISM works correctly, not a real answer. The real
floor needs to be picked from a run against the real file.

Output:
-------
Printed stability sweep for run_value_per_100 specifically, plus a
recommended floor and a properly-filtered output column applying it.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "pitcher_pitch_level_outcomes_2020_2025.csv"
OUTPUT_FILE = "pitcher_pitch_level_outcomes_2020_2025_run_value_fixed.csv"

CANDIDATE_THRESHOLDS = [50, 75, 100, 150, 200, 250, 300]


# ============================================================
# LOAD
# ============================================================

print("Loading pitch-level outcome data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} rows (already MIN_PITCHES=50 filtered for ALL three metrics)")


# ============================================================
# SWEEP THRESHOLDS SPECIFICALLY FOR run_value_per_100
# ============================================================

print("\n\n==============================")
print("Stability sweep: run_value_per_100 specifically")
print("==============================")

results = []
for threshold in CANDIDATE_THRESHOLDS:
    subset = df[df["pitches"] >= threshold]
    if len(subset) < 10:
        continue
    std = subset["run_value_per_100"].std()
    median = subset["run_value_per_100"].median()
    max_abs_from_median = (subset["run_value_per_100"] - median).abs().max()
    results.append({
        "min_pitches": threshold,
        "rows_retained": len(subset),
        "pct_of_original_50_floor_rows": len(subset) / len(df),
        "std": std,
        "max_abs_from_median": max_abs_from_median,
    })

results_df = pd.DataFrame(results)
print(f"\n{results_df.round(3).to_string(index=False)}")


# ============================================================
# PICK AN EVIDENCE-BASED FLOOR
# ============================================================

print("\n\n==============================")
print("Recommended floor")
print("==============================")

# find where max_abs_from_median first drops under a reasonable
# real-world bound for a "per 100 pitches" rate (a single-digit
# number is plausible; double digits are not, for a real, meaningful
# sample)
REASONABLE_MAX_ABS = 6.0

viable = results_df[results_df["max_abs_from_median"] <= REASONABLE_MAX_ABS]
if len(viable) > 0:
    recommended = int(viable.iloc[0]["min_pitches"])
    print(
        f"\nRecommended MIN_PITCHES_RUN_VALUE = {recommended} -- first "
        f"threshold where max_abs_from_median drops to a real-world-"
        f"plausible range (<= {REASONABLE_MAX_ABS})."
    )
else:
    recommended = CANDIDATE_THRESHOLDS[-1]
    print(
        f"\nNo threshold tested reached max_abs_from_median <= "
        f"{REASONABLE_MAX_ABS} -- run_value_per_100 may need an even "
        f"higher floor than tested, or a confidence-shrinkage approach "
        f"instead of a hard cutoff. Using the highest tested threshold "
        f"({recommended}) as a conservative floor for now."
    )


# ============================================================
# APPLY THE FIX -- separate, stricter floor for THIS metric only
# ============================================================

print(f"\nApplying MIN_PITCHES_RUN_VALUE={recommended} to run_value_per_100 specifically "
      f"(xwoba_against and whiff_pct keep their existing 50-pitch floor, already confirmed adequate)...")

df_fixed = df.copy()
n_before = df_fixed["run_value_per_100"].notna().sum()
df_fixed.loc[df_fixed["pitches"] < recommended, "run_value_per_100"] = np.nan
n_after = df_fixed["run_value_per_100"].notna().sum()

print(f"run_value_per_100: {n_before:,} -> {n_after:,} valid values "
      f"({n_after/n_before:.1%} retained) -- xwoba_against and whiff_pct unaffected")

df_fixed.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
