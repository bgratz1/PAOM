"""
92_paom_comparison_scaled.py

Purpose:
--------
Combines 90 and 91 into one script, scaled up: a much larger sample,
multiple years, and -- critically -- a proper chance baseline per
pitcher, not just a raw top-3 percentage.

WHY THE CHANCE BASELINE MATTERS: a small first run showed ADD's top-3
rate at 89.5% with an average of ~4.5 real candidates per pitcher.
If PAOM's picks were pure random noise, landing in the top 3 would
already happen roughly 3/4.5 = 67% of the time purely from there
usually being 3+ candidates to sort through. Without adjusting for
this, a raw top-3 rate can look far more impressive than it actually
is. This script computes, for EACH pitcher, their own expected
chance-level top-3 rate (3/n_candidates, capped at 1.0 when n<=3),
and compares PAOM's REAL top-3 rate against that honest baseline --
per change type.

Same crosswalk and PAOM output files as 90/91 -- reused directly.

Output:
-------
Printed rank distributions, real top-3 rates, and chance-adjusted
comparison, per change type, aggregated across years and a much
larger sample. No file saved -- this is an investigation, not a
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

YEARS = [2021, 2022, 2023, 2024, 2025]  # 2020 excluded -- its much
                                          # smaller PAOM population
                                          # (209 pitchers) would
                                          # otherwise dominate or
                                          # skew a pooled sample
N_SAMPLE_PER_YEAR = 60


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
# CROSSWALK
# ============================================================

print("\nBuilding real player_id <-> PAOM-style name crosswalk...")

chadwick = pybaseball.chadwick_register().dropna(subset=["key_mlbam"])
chadwick["key_mlbam"] = chadwick["key_mlbam"].astype(int)
chadwick["paom_style_name"] = chadwick["name_last"] + ", " + chadwick["name_first"]
id_to_paom_name = dict(zip(chadwick["key_mlbam"], chadwick["paom_style_name"]))

print(f"Crosswalk built for {len(id_to_paom_name):,} real MLBAM player_ids")


# ============================================================
# COMPARISON FUNCTION -- generalized across all four change types
# ============================================================

def run_comparison_for_year(year, change_type, paom_top_df, pitch_type_col):
    target_pool = similarity_features[
        similarity_features["season"] == year
    ].dropna(subset=SIMILARITY_FEATURE_COLS)

    sample = target_pool.sample(n=min(N_SAMPLE_PER_YEAR, len(target_pool)), random_state=42)

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

        type_recs = recommendations[recommendations["change_type"] == change_type].reset_index(drop=True)
        if len(type_recs) == 0:
            continue

        n_candidates = len(type_recs)
        chance_top3_rate = min(3 / n_candidates, 1.0)

        match = type_recs[type_recs["pitch_type"] == paom_pick]
        if len(match) == 0:
            results.append({
                "year": year, "player_id": player_id, "paom_pick": paom_pick,
                "historical_rank": None, "n_historical_candidates": n_candidates,
                "chance_top3_rate": chance_top3_rate, "not_a_viable_candidate": True,
            })
            continue

        historical_rank = match.index[0] + 1
        results.append({
            "year": year, "player_id": player_id, "paom_pick": paom_pick,
            "historical_rank": historical_rank, "n_historical_candidates": n_candidates,
            "chance_top3_rate": chance_top3_rate, "not_a_viable_candidate": False,
        })

    return pd.DataFrame(results)


def summarize(all_results_df, label):
    print(f"\n\n==============================")
    print(f"{label} -- aggregated across {all_results_df['year'].nunique()} years")
    print(f"==============================")

    if len(all_results_df) == 0:
        print("No successful comparisons.")
        return

    not_viable = all_results_df["not_a_viable_candidate"].sum()
    print(f"\n{len(all_results_df):,} total pitcher-seasons compared; "
          f"PAOM's pick NOT viable in {not_viable:,} ({not_viable/len(all_results_df):.1%})")

    viable = all_results_df[~all_results_df["not_a_viable_candidate"]]
    if len(viable) == 0:
        return

    real_top3_rate = (viable["historical_rank"] <= 3).mean()
    expected_chance_rate = viable["chance_top3_rate"].mean()

    print(f"\nReal top-3 rate: {real_top3_rate:.1%}")
    print(f"Expected top-3 rate from PURE CHANCE (adjusted for each pitcher's own "
          f"real candidate count): {expected_chance_rate:.1%}")
    print(f"Lift over chance: {real_top3_rate - expected_chance_rate:+.1%}")

    print(f"\nMean real candidates per pitcher: {viable['n_historical_candidates'].mean():.1f}")

    print(f"\nRank distribution:")
    print(viable["historical_rank"].value_counts().sort_index())

    if real_top3_rate - expected_chance_rate > 0.15:
        print(f"\nVerdict: MEANINGFUL agreement -- real top-3 rate clears the "
              f"chance baseline by a wide margin.")
    elif real_top3_rate - expected_chance_rate > 0.05:
        print(f"\nVerdict: MODEST agreement -- real, but not dramatic, lift over chance.")
    else:
        print(f"\nVerdict: WEAK OR NO real agreement beyond what pure chance would "
              f"already predict given how many candidates these pitchers typically had.")


# ============================================================
# RUN ADD (has its own clean rank/score already)
# ============================================================

add_all = []
for year in YEARS:
    try:
        paom_add = pd.read_csv(f"PAOM_add_recommendations_{year}.csv")
    except FileNotFoundError:
        print(f"\nPAOM_add_recommendations_{year}.csv not found -- skipping {year} for ADD")
        continue
    paom_top = paom_add[paom_add["rank_within_pitcher"] == 1]
    year_results = run_comparison_for_year(year, "ADD", paom_top, "candidate_pitch_type")
    add_all.append(year_results)

if add_all:
    summarize(pd.concat(add_all, ignore_index=True), "ADD")


# ============================================================
# RUN DROP, USAGE_INCREASE, USAGE_DECREASE (constructed rank/score)
# ============================================================

drop_all, increase_all, decrease_all = [], [], []

for year in YEARS:
    try:
        paom_drop = pd.read_csv(f"PAOM_drop_recommendations_{year}.csv")
        paom_usage = pd.read_csv(f"PAOM_usage_recommendations_{year}.csv")
    except FileNotFoundError:
        print(f"\nPAOM drop/usage files not found for {year} -- skipping")
        continue

    paom_drop["drop_score"] = (1 - paom_drop["effectiveness_pctl"]) + (1 - paom_drop["redundancy_pctl"])
    paom_drop["drop_rank"] = paom_drop.groupby("player_name")["drop_score"].rank(ascending=False, method="first")
    paom_drop_top = paom_drop[(paom_drop["drop_candidate"] == True) & (paom_drop["drop_rank"] == 1)]
    drop_all.append(run_comparison_for_year(year, "DROP", paom_drop_top, "pitch_type"))

    usage_increase = paom_usage[paom_usage["recommendation"] == "increase usage"].copy()
    usage_increase["usage_rank"] = usage_increase.groupby("player_name")["pctl_gap"].rank(ascending=False, method="first")
    usage_increase_top = usage_increase[usage_increase["usage_rank"] == 1]
    increase_all.append(run_comparison_for_year(year, "USAGE_INCREASE", usage_increase_top, "pitch_type"))

    usage_decrease = paom_usage[paom_usage["recommendation"] == "decrease usage"].copy()
    usage_decrease["usage_rank"] = usage_decrease.groupby("player_name")["pctl_gap"].rank(ascending=True, method="first")
    usage_decrease_top = usage_decrease[usage_decrease["usage_rank"] == 1]
    decrease_all.append(run_comparison_for_year(year, "USAGE_DECREASE", usage_decrease_top, "pitch_type"))

if drop_all:
    summarize(pd.concat(drop_all, ignore_index=True), "DROP")
if increase_all:
    summarize(pd.concat(increase_all, ignore_index=True), "USAGE_INCREASE")
if decrease_all:
    summarize(pd.concat(decrease_all, ignore_index=True), "USAGE_DECREASE")
