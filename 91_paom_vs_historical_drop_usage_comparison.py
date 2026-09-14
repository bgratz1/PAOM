"""
91_paom_vs_historical_drop_usage_comparison.py

Purpose:
--------
Extends 90's comparison to DROP and USAGE_INCREASE/USAGE_DECREASE --
the two PAOM outputs that don't have a clean, pre-built numeric
rank the way ADD does.

CONSTRUCTED RANKING SIGNALS, since PAOM doesn't provide these
natively:
- DROP: PAOM's drop_candidate is a boolean flag, not a ranked list.
  A continuous drop_score is constructed as (1 - effectiveness_pctl)
  + (1 - redundancy_pctl) -- higher means both less effective AND
  more redundant, i.e. a stronger real drop candidate under PAOM's
  own stated logic (low effectiveness AND redundant, not either
  alone -- see 16's docstring). Ranked descending per pitcher.
- USAGE: PAOM's recommendation is a three-way category (increase/
  decrease/maintain), with pctl_gap as the underlying continuous
  signal. For USAGE_INCREASE comparison, only PAOM's "increase
  usage" rows are used, ranked by pctl_gap descending (strongest
  increase signal first). Same for USAGE_DECREASE, ranked by
  pctl_gap ascending (most negative = strongest decrease signal).

Same crosswalk and rank-lookup mechanics as 90 -- reused directly,
not reconstructed.

Output:
-------
Printed rank-position distributions for DROP, USAGE_INCREASE, and
USAGE_DECREASE separately. No file saved -- investigation, not a
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

YEAR = 2024
N_SAMPLE = 30

PAOM_DROP_FILE = f"PAOM_drop_recommendations_{YEAR}.csv"
PAOM_USAGE_FILE = f"PAOM_usage_recommendations_{YEAR}.csv"


# ============================================================
# IMPORT FROM 45
# ============================================================

print("Loading recommendation engine (45)...")

spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine", "45_similarity_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(spec)
sys.modules["similarity_recommendation_engine"] = rec_engine
spec.loader.exec_module(rec_engine)

recommend_arsenal_changes = rec_engine.recommend_arsenal_changes
similarity_features = rec_engine.similarity_features
SIMILARITY_FEATURE_COLS = rec_engine.SIMILARITY_FEATURE_COLS


# ============================================================
# CROSSWALK -- same as 90
# ============================================================

print("\nBuilding real player_id <-> PAOM-style name crosswalk...")

chadwick = pybaseball.chadwick_register().dropna(subset=["key_mlbam"])
chadwick["key_mlbam"] = chadwick["key_mlbam"].astype(int)
chadwick["paom_style_name"] = chadwick["name_last"] + ", " + chadwick["name_first"]
id_to_paom_name = dict(zip(chadwick["key_mlbam"], chadwick["paom_style_name"]))

print(f"Crosswalk built for {len(id_to_paom_name):,} real MLBAM player_ids")


# ============================================================
# LOAD PAOM's DROP AND USAGE RECOMMENDATIONS
# ============================================================

print(f"\nLoading {PAOM_DROP_FILE} and {PAOM_USAGE_FILE}...")

paom_drop = pd.read_csv(PAOM_DROP_FILE)
paom_usage = pd.read_csv(PAOM_USAGE_FILE)

# construct DROP's ranking signal (see docstring)
paom_drop["drop_score"] = (
    (1 - paom_drop["effectiveness_pctl"]) + (1 - paom_drop["redundancy_pctl"])
)
paom_drop["drop_rank"] = (
    paom_drop.groupby("player_name")["drop_score"]
    .rank(ascending=False, method="first")
)
paom_drop_top = paom_drop[
    (paom_drop["drop_candidate"] == True) & (paom_drop["drop_rank"] == 1)
]

# USAGE_INCREASE and USAGE_DECREASE ranked separately, per PAOM's
# own categorical split
usage_increase = paom_usage[paom_usage["recommendation"] == "increase usage"].copy()
usage_increase["usage_rank"] = (
    usage_increase.groupby("player_name")["pctl_gap"]
    .rank(ascending=False, method="first")
)
usage_increase_top = usage_increase[usage_increase["usage_rank"] == 1]

usage_decrease = paom_usage[paom_usage["recommendation"] == "decrease usage"].copy()
usage_decrease["usage_rank"] = (
    usage_decrease.groupby("player_name")["pctl_gap"]
    .rank(ascending=True, method="first")
)
usage_decrease_top = usage_decrease[usage_decrease["usage_rank"] == 1]

print(f"PAOM has a #1 DROP pick for {len(paom_drop_top):,} pitchers")
print(f"PAOM has a #1 USAGE_INCREASE pick for {len(usage_increase_top):,} pitchers")
print(f"PAOM has a #1 USAGE_DECREASE pick for {len(usage_decrease_top):,} pitchers")


# ============================================================
# SAMPLE AND COMPARE, PER CHANGE TYPE
# ============================================================

target_pool = similarity_features[
    similarity_features["season"] == YEAR
].dropna(subset=SIMILARITY_FEATURE_COLS)

sample = target_pool.sample(n=min(N_SAMPLE, len(target_pool)), random_state=42)


def run_comparison(paom_top_df, pitch_type_col, change_type_label, historical_change_type):
    results = []
    for _, target_row in sample.iterrows():
        player_id = target_row["player_id"]
        season = target_row["season"]

        paom_name = id_to_paom_name.get(player_id)
        if paom_name is None:
            continue

        paom_match = paom_top_df[paom_top_df["player_name"] == paom_name]
        if len(paom_match) == 0:
            continue

        paom_pick = paom_match.iloc[0][pitch_type_col]

        try:
            recommendations, _ = recommend_arsenal_changes(player_id, season, outcome_metric="xwoba_against")
        except Exception:
            continue

        type_recs = recommendations[
            recommendations["change_type"] == historical_change_type
        ].reset_index(drop=True)
        if len(type_recs) == 0:
            continue

        match = type_recs[type_recs["pitch_type"] == paom_pick]
        if len(match) == 0:
            results.append({
                "player_id": player_id, "paom_pick": paom_pick,
                "historical_rank": None, "n_historical_candidates": len(type_recs),
                "not_a_viable_candidate": True,
            })
            continue

        historical_rank = match.index[0] + 1
        results.append({
            "player_id": player_id, "paom_pick": paom_pick,
            "historical_rank": historical_rank, "n_historical_candidates": len(type_recs),
            "not_a_viable_candidate": False,
        })

    results_df = pd.DataFrame(results)

    print(f"\n\n==============================")
    print(f"{change_type_label}")
    print(f"==============================")

    if len(results_df) == 0:
        print("No successful comparisons.")
        return results_df

    not_viable = results_df["not_a_viable_candidate"].sum()
    print(f"\n{len(results_df)} pitchers compared; PAOM's pick NOT viable "
          f"in {not_viable} of them")

    viable = results_df[~results_df["not_a_viable_candidate"]]
    if len(viable) > 0:
        top_3_rate = (viable["historical_rank"] <= 3).mean()
        print(f"PAOM's #1 pick landed in the historical engine's own TOP 3: {top_3_rate:.1%}")
        print(f"\nRank distribution:")
        print(viable["historical_rank"].value_counts().sort_index())

    return results_df


drop_results = run_comparison(paom_drop_top, "pitch_type", "DROP comparison", "DROP")
increase_results = run_comparison(usage_increase_top, "pitch_type", "USAGE_INCREASE comparison", "USAGE_INCREASE")
decrease_results = run_comparison(usage_decrease_top, "pitch_type", "USAGE_DECREASE comparison", "USAGE_DECREASE")
