"""
33_outcome_stability_diagnostic.py

Purpose:
--------
32_pull_pitcher_season_metrics.py's MIN_BIP=20 filter was picked as
a reasonable-sounding guess (same honest caveat every cutoff in this
project started with) specifically to exclude position-player mop-up
appearances. A real run surfaced a case sitting right at that floor
(Oliver Perez, 2022: BIP=20, FIP=7.37, ERA_plus=27.9) that looks like
small-sample noise, not a real reflection of talent -- the same kind
of instability this project has repeatedly found and fixed elsewhere
(confidence shrinkage, Effectiveness's cutoff sweeps).

This checks whether that's an isolated case or a real, widespread
pattern, across all three outcome metrics (FIP, xwOBA-against, WAR)
-- not just FIP, since all three are computed from the same thin
samples and could share the same instability.

Three checks:
1. Bucket pitcher-seasons by BIP, show each bucket's outcome-metric
   spread (std, min, max) -- if low-BIP buckets show dramatically
   wider spread than high-BIP buckets, that's direct evidence of
   instability at low sample sizes, not just a guess.
2. List the most extreme outcome values league-wide, alongside their
   BIP -- lets you see concretely how many "Perez-2022-style" cases
   exist and whether they cluster at low BIP.
3. Sweep candidate MIN_BIP thresholds, reporting rows/pitchers
   retained (the coverage cost) against the resulting outcome
   spread (the stability benefit) at each -- the same coverage-vs-
   stability tradeoff already used throughout this project's cutoff
   work, applied here to pick an evidence-based threshold instead of
   another guess.

Output:
-------
Printed diagnostics only -- no file saved. Use the results to pick
a new MIN_BIP value for 32_pull_pitcher_season_metrics.py.
"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "pitcher_season_metrics_2020_2025.csv"

BIP_BUCKETS = [20, 30, 50, 75, 100, 150, 200, 1000]  # bucket edges
CANDIDATE_MIN_BIP = [20, 30, 40, 50, 75, 100, 150]

N_EXTREME_TO_SHOW = 15


# ============================================================
# LOAD
# ============================================================

print("Loading pitcher season metrics...")

df = pd.read_csv(INPUT_FILE)

required_cols = ["player_id", "season", "name", "bip", "fip", "xwoba_against", "war"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}. This script needs "
        f"32_pull_pitcher_season_metrics.py's output with bip, fip, "
        f"xwoba_against, and war all present."
    )

print(f"Loaded {len(df):,} pitcher-season rows\n")


# ============================================================
# CHECK 1: OUTCOME SPREAD BY BIP BUCKET
# ============================================================

print("==============================")
print("CHECK 1: Outcome spread by BIP bucket")
print("==============================")
print(
    "If low-BIP buckets show dramatically wider spread (std, min,\n"
    "max) than high-BIP buckets, that's direct evidence of small-\n"
    "sample instability, not just a hunch from one example.\n"
)

df["bip_bucket"] = pd.cut(df["bip"], bins=BIP_BUCKETS, right=False)

for metric in ["fip", "xwoba_against", "war"]:
    print(f"\n--- {metric} ---")
    bucket_stats = df.groupby("bip_bucket", observed=True)[metric].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(3)
    print(bucket_stats)


# ============================================================
# CHECK 2: MOST EXTREME OUTCOME VALUES, WITH THEIR BIP
# ============================================================

print("\n\n==============================")
print("CHECK 2: Most extreme outcome values (top and bottom), with BIP")
print("==============================")

for metric in ["fip", "xwoba_against", "war"]:
    print(f"\n--- {metric}: highest values ---")
    top = df.nlargest(N_EXTREME_TO_SHOW, metric)[["name", "season", "bip", metric]]
    print(top.to_string(index=False))

    print(f"\n--- {metric}: lowest values ---")
    bottom = df.nsmallest(N_EXTREME_TO_SHOW, metric)[["name", "season", "bip", metric]]
    print(bottom.to_string(index=False))

    extreme_combined = pd.concat([top, bottom])
    median_bip = df["bip"].median()
    extreme_median_bip = extreme_combined["bip"].median()
    print(
        f"\nMedian BIP among these {metric} extremes: {extreme_median_bip:.0f} "
        f"(population median BIP: {median_bip:.0f}) -- "
        f"{'extremes skew toward LOW BIP, consistent with instability' if extreme_median_bip < median_bip else 'extremes do NOT obviously skew toward low BIP'}"
    )


# ============================================================
# CHECK 3: SWEEP CANDIDATE MIN_BIP THRESHOLDS
# ============================================================

print("\n\n==============================")
print("CHECK 3: Candidate MIN_BIP thresholds -- coverage vs. stability")
print("==============================")

results = []
for threshold in CANDIDATE_MIN_BIP:
    subset = df[df["bip"] >= threshold]
    row = {
        "min_bip": threshold,
        "rows_retained": len(subset),
        "pitchers_retained": subset["player_id"].nunique(),
        "pct_of_rows_retained": len(subset) / len(df),
    }
    for metric in ["fip", "xwoba_against", "war"]:
        row[f"{metric}_std"] = subset[metric].std()
        row[f"{metric}_max_abs_from_median"] = (
            (subset[metric] - subset[metric].median()).abs().max()
        )
    results.append(row)

results_df = pd.DataFrame(results)
print(results_df.round(3).to_string(index=False))

print(
    "\nLook for where the *_std and *_max_abs_from_median columns "
    "stop dropping meaningfully as min_bip increases -- that's "
    "roughly where real instability ends and you're just trading "
    "away coverage for no further stability benefit. rows_retained/ "
    "pct_of_rows_retained show the coverage cost of each candidate."
)
