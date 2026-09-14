"""
90_paom_vs_historical_add_comparison.py

Purpose:
--------
First real comparison in the PAOM-vs-historical-engine investigation
-- ADD only (the cleanest, most direct mapping: PAOM's ADD output
already has a numeric add_score and explicit rank_within_pitcher,
directly comparable to the historical engine's own ranked ADD
candidates for the same pitcher).

For real pitcher-seasons: get PAOM's #1-ranked ADD pick, find where
that SAME pitch type lands on the historical engine's own ranked
list for that same pitcher. A consistently high rank position (near
the top of the historical engine's own list) means the two systems
tend to agree on what the best ADD choice is; a scattered
distribution means they disagree.

Uses pybaseball's real Chadwick register (the same crosswalk 46
already used) to match PAOM's player_name ("Last, First") to the
MLBAM player_id the historical engine's own data is keyed on.

Output:
-------
Printed rank-position distribution, plus a verdict on agreement.
No file saved -- this is an investigation, not a deliverable.
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

YEAR = 2024  # PAOM recommendation file to compare against
N_SAMPLE = 30

PAOM_ADD_FILE = f"PAOM_add_recommendations_{YEAR}.csv"


# ============================================================
# IMPORT FROM 45 (cascades through 38)
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
# BUILD REAL NAME <-> MLBAM ID CROSSWALK
# ============================================================

print("\nBuilding real player_id <-> PAOM-style name crosswalk (Chadwick register)...")

chadwick = pybaseball.chadwick_register()
chadwick = chadwick.dropna(subset=["key_mlbam"])
chadwick["key_mlbam"] = chadwick["key_mlbam"].astype(int)

# PAOM's player_name format is "Last, First" -- build the same
# string so it can be matched directly against PAOM's own output
chadwick["paom_style_name"] = chadwick["name_last"] + ", " + chadwick["name_first"]

id_to_paom_name = dict(zip(chadwick["key_mlbam"], chadwick["paom_style_name"]))

print(f"Crosswalk built for {len(id_to_paom_name):,} real MLBAM player_ids")


# ============================================================
# LOAD PAOM's ADD RECOMMENDATIONS
# ============================================================

print(f"\nLoading {PAOM_ADD_FILE}...")

try:
    paom_add = pd.read_csv(PAOM_ADD_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"{PAOM_ADD_FILE} not found. This script needs your REAL "
        f"PAOM add recommendations file for {YEAR} (from 88) -- the "
        f"crosswalk mechanics above are confirmed working with real "
        f"data, but the actual comparison needs your real PAOM output."
    )

# PAOM's #1 pick per pitcher only, for this first, simplest comparison
paom_top_pick = paom_add[paom_add["rank_within_pitcher"] == 1].copy()
print(f"PAOM has a #1 ADD pick for {len(paom_top_pick):,} pitchers in {YEAR}")


# ============================================================
# SAMPLE REAL PITCHER-SEASONS AND RUN THE COMPARISON
# ============================================================

print(f"\nSampling {N_SAMPLE} real {YEAR} pitcher-seasons...")

target_pool = similarity_features[
    (similarity_features["season"] == YEAR)
].dropna(subset=SIMILARITY_FEATURE_COLS)

sample = target_pool.sample(n=min(N_SAMPLE, len(target_pool)), random_state=42)

results = []

for _, target_row in sample.iterrows():
    player_id = target_row["player_id"]
    season = target_row["season"]

    paom_name = id_to_paom_name.get(player_id)
    if paom_name is None:
        continue

    paom_match = paom_top_pick[paom_top_pick["player_name"] == paom_name]
    if len(paom_match) == 0:
        continue

    paom_pick = paom_match.iloc[0]["candidate_pitch_type"]

    try:
        recommendations, _ = recommend_arsenal_changes(player_id, season, outcome_metric="xwoba_against")
    except Exception:
        continue

    add_recs = recommendations[recommendations["change_type"] == "ADD"].reset_index(drop=True)
    if len(add_recs) == 0:
        continue

    match = add_recs[add_recs["pitch_type"] == paom_pick]
    if len(match) == 0:
        # PAOM's pick wasn't even a viable candidate in the historical
        # engine's own search -- worth recording as its own outcome,
        # not silently dropped
        results.append({
            "player_id": player_id,
            "paom_pick": paom_pick,
            "historical_rank": None,
            "n_historical_candidates": len(add_recs),
            "not_a_viable_candidate": True,
        })
        continue

    historical_rank = match.index[0] + 1  # 1-indexed rank position

    results.append({
        "player_id": player_id,
        "paom_pick": paom_pick,
        "historical_rank": historical_rank,
        "n_historical_candidates": len(add_recs),
        "not_a_viable_candidate": False,
    })

results_df = pd.DataFrame(results)

print(f"\n{len(results_df):,} pitchers successfully compared")


# ============================================================
# RESULTS
# ============================================================

print("\n\n==============================")
print("Where does PAOM's #1 ADD pick land on the historical engine's own ranking?")
print("==============================")

if len(results_df) == 0:
    print("\nNo successful comparisons -- check that the crosswalk matched real names, "
          "and that PAOM_add_recommendations file covers this year's real pitchers.")
else:
    viable = results_df[~results_df["not_a_viable_candidate"]]
    not_viable_count = results_df["not_a_viable_candidate"].sum()

    print(f"\nPAOM's pick was NOT a viable candidate in the historical engine's own "
          f"search for {not_viable_count} of {len(results_df)} pitchers "
          f"(no real historical precedent for that specific change).")

    if len(viable) > 0:
        print(f"\nFor the {len(viable)} pitchers where it WAS viable:")
        print(viable[["historical_rank", "n_historical_candidates"]].describe())

        top_3_rate = (viable["historical_rank"] <= 3).mean()
        print(f"\nPAOM's #1 pick landed in the historical engine's own TOP 3: "
              f"{top_3_rate:.1%} of the time")

        print(f"\nFull rank distribution:")
        print(viable["historical_rank"].value_counts().sort_index())

print("\nSample results:")
print(results_df.to_string(index=False))
