"""
46_integrate_with_paom_engine.py

Purpose:
--------
Bridges the ORIGINAL PAOM pipeline's 16_recommendation_engine.py
(current-2025-snapshot-based ADD recommendations, keyed by
player_name) with this whole Kaggle-based thread's historical-
precedent engine (38/45, keyed by MLBAM player_id) -- enriching
16's existing "add this pitch type" suggestions with a predicted
magnitude and real historical comps, rather than replacing them
with an entirely separate, disconnected recommendation.

THE REAL INTEGRATION OBSTACLE: 16's output is keyed by player_name
in "Last, First" format (the original PAOM pipeline's convention) --
this whole Kaggle sub-thread is keyed by player_id (MLBAM numeric
ID). This project already learned, early on, that name-matching
across sources is unreliable (accents, "Jr." suffixes). Rather than
risk that again, this uses pybaseball's player_search_list() -- the
same Chadwick-register crosswalk tool already used successfully in
32_pull_pitcher_season_metrics.py -- as a proper ID bridge instead
of fuzzy name matching.

DESIGN: for each row in 16's ADD_OUTPUT (a specific pitcher +
specific candidate pitch type 16 already suggested), this does NOT
run 45's full candidate search from scratch -- it directly evaluates
THAT SPECIFIC suggestion using 38's predict_arsenal_change_effect()
and 45's estimate_realistic_usage_delta(), enriching 16's own pick
with a predicted magnitude, sample size, and top historical comps.
This answers "how well-supported is 16's existing suggestion by
real historical precedent," not "what would this separate engine
have picked instead."

DISAMBIGUATION: a name can match multiple real players in the
Chadwick register (common last names). Matches are narrowed using
mlb_played_last -- preferring players whose career plausibly extends
through the target season (2025, since that's what 16's pipeline is
built on). Any name that's unmatched or remains ambiguous after this
is EXCLUDED and reported explicitly, not guessed at silently.

Output:
-------
PAOM_add_recommendations_enriched.csv -- 16's original ADD
recommendations with predicted_change, n_historical_events, method,
and a compact top-comps string appended where a match was found.
"""

import importlib.util
import sys

import pandas as pd
import numpy as np

import pybaseball

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FROM 45 (which itself imports from 38)
# ============================================================

print("Loading training data and prediction functions from 45...")

spec = importlib.util.spec_from_file_location(
    "similarity_recommendation_engine", "45_similarity_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(spec)
sys.modules["similarity_recommendation_engine"] = rec_engine
spec.loader.exec_module(rec_engine)

similarity_features = rec_engine.similarity_features
outcomes = rec_engine.outcomes
predict_arsenal_change_effect = rec_engine.predict_arsenal_change_effect
estimate_realistic_usage_delta = rec_engine.estimate_realistic_usage_delta
show_similar_comps = rec_engine.show_similar_comps
SIMILARITY_FEATURE_COLS = rec_engine.SIMILARITY_FEATURE_COLS


# ============================================================
# SETTINGS
# ============================================================

ADD_RECOMMENDATIONS_FILE = "PAOM_add_recommendations.csv"
OUTPUT_FILE = "PAOM_add_recommendations_enriched.csv"

TARGET_SEASON = 2025  # 16_recommendation_engine.py's pipeline is
                        # built on 2025 data -- this pins the
                        # crosswalk disambiguation and the Kaggle-
                        # side lookups to that same season
OUTCOME_METRIC = "xwoba_against"  # best-performing metric, per 41

TOP_K_COMPS_STRING = 3  # kept compact for a single enriched-CSV column


# ============================================================
# LOAD 16's ADD RECOMMENDATIONS
# ============================================================

print(f"\nLoading {ADD_RECOMMENDATIONS_FILE}...")

add_recs = pd.read_csv(ADD_RECOMMENDATIONS_FILE)

print(f"Loaded {len(add_recs):,} rows, {add_recs['player_name'].nunique():,} unique pitchers")


# ============================================================
# CROSSWALK player_name -> MLBAM player_id
# ============================================================

print("\nCrosswalking player names to MLBAM IDs (Chadwick register)...")

def parse_last_first(name):
    """
    '16's convention: "Last, First" (e.g. "Skenes, Paul"). Splits on
    the FIRST comma only -- suffixes like "McCullers Jr., Lance"
    keep "Jr." attached to the last-name portion, matching how the
    original PAOM pipeline's own name field is structured. This is a
    real, imperfect heuristic, not a guaranteed-correct parser --
    unmatched results are reported explicitly below, not silently
    accepted.
    """
    if "," not in name:
        return None, None
    last, first = name.split(",", 1)
    return last.strip(), first.strip()

unique_names = add_recs["player_name"].unique()
name_pairs = []
for name in unique_names:
    last, first = parse_last_first(name)
    if last is not None:
        name_pairs.append((last, first))

# NOTE: pybaseball.player_search_list() is BROKEN on any modern
# pandas (2.0+) -- its internal loop calls results.append(...),
# using the DataFrame.append() method removed in pandas 2.0.
# Confirmed directly: the underlying register download succeeds
# (GitHub access works fine in this environment), but search_list's
# own aggregation step crashes with AttributeError. This is a bug in
# the pybaseball library itself, not fixable from this script's side
# by adjusting how it's called.
#
# FIX: playerid_lookup() (the single-name lookup search_list() is
# SUPPOSED to just loop) does NOT have this bug -- it uses .loc[]
# filtering, no .append() anywhere. The underlying player register
# is a module-level singleton (_get_client()), downloaded once and
# cached -- so looping playerid_lookup() here does NOT re-download
# the full register for every name, only the first call pays that
# cost. This is a correct, direct reimplementation of what
# search_list() was supposed to do, not a workaround.
crosswalk_frames = []
for last, first in name_pairs:
    try:
        result = pybaseball.playerid_lookup(last, first)
        if len(result) > 0:
            crosswalk_frames.append(result)
    except Exception as e:
        print(f"  lookup failed for ({last}, {first}): {e}")

crosswalk_raw = pd.concat(crosswalk_frames, ignore_index=True) if crosswalk_frames else pd.DataFrame()

print(f"Crosswalk returned {len(crosswalk_raw):,} rows for {len(name_pairs):,} queried names")
print(f"Crosswalk columns: {crosswalk_raw.columns.tolist()}")

# defensive column detection -- confirmed columns from this
# project's earlier reverse-lookup usage (32_pull_pitcher_season_
# metrics.py), but not independently re-verified live for THIS
# specific forward-search function, since no live network access
# was available to test it directly
mlbam_col = next((c for c in crosswalk_raw.columns if c.lower() in ("key_mlbam", "mlbam_id")), None)
last_col = next((c for c in crosswalk_raw.columns if c.lower() in ("name_last",)), None)
first_col = next((c for c in crosswalk_raw.columns if c.lower() in ("name_first",)), None)
played_last_col = next((c for c in crosswalk_raw.columns if "played_last" in c.lower()), None)

if mlbam_col is None or last_col is None or first_col is None:
    raise ValueError(
        f"Could not identify expected columns in crosswalk result: "
        f"{crosswalk_raw.columns.tolist()}. Inspect manually and "
        f"adjust the column detection above."
    )

print(f"Using columns: mlbam='{mlbam_col}', last='{last_col}', first='{first_col}', played_last='{played_last_col}'")


# ============================================================
# DISAMBIGUATE AND BUILD name -> player_id MAP
# ============================================================

name_to_id = {}
n_unmatched = 0
n_ambiguous_resolved = 0
n_ambiguous_unresolved = 0

for name in unique_names:
    last, first = parse_last_first(name)
    if last is None:
        n_unmatched += 1
        continue

    matches = crosswalk_raw[
        (crosswalk_raw[last_col].str.lower() == last.lower())
        & (crosswalk_raw[first_col].str.lower() == first.lower())
    ]

    if len(matches) == 0:
        n_unmatched += 1
        continue

    if len(matches) == 1:
        name_to_id[name] = matches.iloc[0][mlbam_col]
        continue

    # multiple matches -- disambiguate by preferring a career that
    # plausibly extends through TARGET_SEASON (still active, or an
    # end year at/after it)
    if played_last_col is not None:
        plausible = matches[
            matches[played_last_col].isna()
            | (pd.to_numeric(matches[played_last_col], errors="coerce") >= TARGET_SEASON - 1)
        ]
        if len(plausible) == 1:
            name_to_id[name] = plausible.iloc[0][mlbam_col]
            n_ambiguous_resolved += 1
            continue

    n_ambiguous_unresolved += 1

print(
    f"\nMatched: {len(name_to_id):,} of {len(unique_names):,} unique names "
    f"({n_ambiguous_resolved:,} resolved from multiple candidates)"
)
print(
    f"Unmatched: {n_unmatched:,}, still ambiguous after disambiguation: "
    f"{n_ambiguous_unresolved:,} -- these are EXCLUDED below, not guessed at"
)

add_recs["player_id"] = add_recs["player_name"].map(name_to_id)


# ============================================================
# ENRICH EACH ROW WITH HISTORICAL-PRECEDENT EVIDENCE
# ============================================================

print("\nEnriching each recommendation with historical-precedent evidence...")

enriched_rows = []
n_no_similarity_profile = 0
n_no_outcome_baseline = 0
n_enriched = 0

for _, row in add_recs.iterrows():
    player_id = row.get("player_id")
    pitch_type = row["candidate_pitch_type"]

    enrichment = {
        "predicted_change": np.nan,
        "n_historical_events": np.nan,
        "method": "not_enriched",
        "top_comps": "",
    }

    if pd.notna(player_id):
        sim_rows = similarity_features[
            (similarity_features["player_id"] == player_id)
            & (similarity_features["season"] == TARGET_SEASON)
        ]
        outcome_rows = outcomes[outcomes["player_id"] == player_id]

        if len(sim_rows) == 0:
            n_no_similarity_profile += 1
        elif len(outcome_rows) == 0 or OUTCOME_METRIC not in outcome_rows.columns:
            n_no_outcome_baseline += 1
        else:
            sim_row = sim_rows.iloc[0]
            target_features = {c: sim_row[c] for c in SIMILARITY_FEATURE_COLS}

            outcome_rows = outcome_rows.sort_values("season")
            target_outcome_before = outcome_rows[OUTCOME_METRIC].iloc[-1]

            realistic_delta, _ = estimate_realistic_usage_delta(
                target_features, pitch_type, "ADD", player_id,
                TARGET_SEASON, target_outcome_before, OUTCOME_METRIC
            )

            if realistic_delta is not None:
                final_features = dict(target_features)
                final_features["usage_pct_delta"] = realistic_delta

                result = predict_arsenal_change_effect(
                    final_features, pitch_type, "ADD",
                    target_outcome_before, OUTCOME_METRIC,
                    player_id, TARGET_SEASON
                )

                if result["predicted_outcome"] is not None:
                    top_comps = ""
                    if "combined_weight" in result["group"].columns:
                        top = result["group"].nlargest(TOP_K_COMPS_STRING, "combined_weight")
                        top_comps = "; ".join(top["player_name"].tolist())

                    enrichment = {
                        "predicted_change": result["predicted_change"],
                        "n_historical_events": result["n_historical_events"],
                        "method": result["method"],
                        "top_comps": top_comps,
                    }
                    n_enriched += 1

    enriched_row = dict(row)
    enriched_row.update(enrichment)
    enriched_rows.append(enriched_row)

enriched_df = pd.DataFrame(enriched_rows)


# ============================================================
# SAVE
# ============================================================

enriched_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n==============================")
print(f"Saved: {OUTPUT_FILE}")
print(f"==============================")
print(f"\nTotal rows: {len(enriched_df):,}")
print(f"Successfully enriched with historical-precedent evidence: {n_enriched:,}")
print(f"Skipped -- no MLBAM match: {(enriched_df['player_id'].isna()).sum():,}")
print(f"Skipped -- matched but no similarity profile for {TARGET_SEASON}: {n_no_similarity_profile:,}")
print(f"Skipped -- matched but no {OUTCOME_METRIC} baseline: {n_no_outcome_baseline:,}")

print(f"\nSample enriched rows:")
sample_cols = ["player_name", "candidate_pitch_type", "predicted_change", "n_historical_events", "top_comps"]
sample_cols = [c for c in sample_cols if c in enriched_df.columns]
print(enriched_df[enriched_df["method"] != "not_enriched"][sample_cols].head(10).to_string(index=False))
